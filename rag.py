import os
import json
import httpx
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

class PlantRAG:
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)
        self.index_json_path = os.path.join(self.base_dir, 'vector_store.json')
        self.disease_csv_path = os.path.abspath(os.path.join(self.base_dir, 'disease_info.csv'))
        self.supplement_csv_path = os.path.abspath(os.path.join(self.base_dir, 'supplement_info.csv'))
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._is_valid_key = bool(self.api_key and self.api_key != "YOUR_GEMINI_API_KEY_HERE" and not self.api_key.startswith("AQ.Ab8RN6LWV"))
        self.documents = []
        self.embeddings = []
        self.faiss_index = None

    def fetch_embedding(self, text: str) -> list[float]:
        """Fetches vector embedding for text using direct HTTP REST API."""
        if not self._is_valid_key:
            return [0.0] * 768  # Return zero vector fallback
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={self.api_key}"
        payload = {
            "model": "models/gemini-embedding-001",
            "content": {
                "parts": [{"text": text}]
            }
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.post(url, json=payload)
                if res.status_code == 200:
                    return res.json()["embedding"]["values"]
                else:
                    print(f"Embedding API error {res.status_code}: {res.text}")
                    if res.status_code in [400, 403, 429]:
                        print(f"Warning: Embedding API returned status {res.status_code}. Using zero-vector fallback for this request.")
        except Exception as e:
            print(f"Error fetching embedding via REST: {e}")
        return [0.0] * 768  # Return zero vector fallback

    def initialize_index(self):
        """Loads index from JSON file if exists, otherwise builds it from CSVs."""
        if os.path.exists(self.index_json_path):
            try:
                with open(self.index_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
                    self.embeddings = data.get("embeddings", [])
                print(f"Loaded existing vector_store.json index successfully ({len(self.documents)} records).")
                self.build_faiss_index()
                return True
            except Exception as e:
                print(f"Failed to load JSON index: {e}. Rebuilding...")

        if not self._is_valid_key:
            print("WARNING: GEMINI_API_KEY is not set or invalid. REST RAG running in offline mode.")
            return False

        return self.build_index()

    def build_index(self):
        """Constructs vector index and saves it to JSON file."""
        try:
            disease_df = pd.read_csv(self.disease_csv_path, encoding='cp1252')
            supplement_df = pd.read_csv(self.supplement_csv_path, encoding='cp1252')
            
            docs = []
            embeds = []
            
            for idx, row in disease_df.iterrows():
                disease_name = row.get('disease_name', 'Unknown Disease')
                desc = row.get('description', '')
                possible_steps = row.get('Possible Steps', '')
                
                supp_name = "N/A"
                buy_link = ""
                if idx < len(supplement_df):
                    supp_name = supplement_df.iloc[idx].get('supplement name', 'N/A')
                    buy_link = supplement_df.iloc[idx].get('buy link', '')
                
                text_content = (
                    f"Plant Disease: {disease_name}\n"
                    f"Description: {desc}\n"
                    f"Symptoms and Treatment Steps: {possible_steps}\n"
                    f"Recommended Supplement: {supp_name}\n"
                    f"Purchase Link: {buy_link}\n"
                )
                
                print(f"Embedding record {idx+1}/{len(disease_df)}: {disease_name}...")
                vector = self.fetch_embedding(text_content)
                
                docs.append({
                    "text": text_content,
                    "metadata": {
                        "disease_name": disease_name,
                        "index": idx,
                        "supplement_name": supp_name,
                        "buy_link": buy_link
                    }
                })
                embeds.append(vector)
            
            self.documents = docs
            self.embeddings = embeds
            
            # Save index
            with open(self.index_json_path, 'w', encoding='utf-8') as f:
                json.dump({"documents": docs, "embeddings": embeds}, f, ensure_ascii=False, indent=2)
                
            print("Vector index built and saved to vector_store.json.")
            self.build_faiss_index()
            return True
        except Exception as e:
            print(f"Error building vector index: {e}")
            import traceback
            traceback.print_exc()
            return False

    def retrieve_context(self, query: str, top_k: int = 2) -> str:
        """Calculates cosine similarity and returns top-k documents. Falls back to text search if API fails."""
        if not self.documents:
            self.initialize_index()
            if not self.documents:
                return "RAG database unavailable."
                
        # To optimize API quota usage and speed up scan latency, we use direct local text matching
        # since the query is always the exact class name predicted by the CNN model.
        return self.fallback_text_search(query, top_k)

    def build_faiss_index(self):
        """Constructs an in-memory FAISS IndexFlatIP for vector similarity search."""
        import faiss
        if not self.embeddings:
            print("No embeddings loaded to build FAISS index.")
            return False
        try:
            embeddings_np = np.array(self.embeddings, dtype='float32')
            faiss.normalize_L2(embeddings_np)
            self.faiss_index = faiss.IndexFlatIP(768)
            self.faiss_index.add(embeddings_np)
            print(f"Built FAISS IndexFlatIP successfully with {self.faiss_index.ntotal} records.")
            return True
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            return False

    def search_vector_index(self, query_text: str, top_k: int = 2) -> str:
        """Runs vector similarity search against the FAISS index. Falls back to text search if offline/error."""
        if not self.documents:
            self.initialize_index()
            
        if self.faiss_index is None:
            self.build_faiss_index()

        if self.faiss_index is not None and self._is_valid_key:
            try:
                # Fetch query vector embedding from Gemini API
                query_vector = self.fetch_embedding(query_text)
                query_vector_np = np.array([query_vector], dtype='float32')
                
                # Check if it's the zero vector fallback (API key limit or offline during call)
                if not np.all(query_vector_np == 0):
                    import faiss
                    faiss.normalize_L2(query_vector_np)
                    similarities, indices = self.faiss_index.search(query_vector_np, top_k)
                    
                    context_blocks = []
                    for idx in indices[0]:
                        if 0 <= idx < len(self.documents):
                            context_blocks.append(self.documents[idx]["text"])
                            
                    if context_blocks:
                        return "\n---\n".join(context_blocks)
            except Exception as e:
                print(f"Error during FAISS similarity search: {e}")

        # Fallback to local text-based matching if offline, error, or rate-limited
        return self.fallback_text_search(query_text, top_k)

    def fallback_text_search(self, query: str, top_k: int = 2) -> str:
        """Fallback search using simple word matching and disease name similarity."""
        if not self.documents:
            return "RAG database is empty."
            
        # Clean query: lowercase, replace underscores and special characters
        query_clean = query.lower().replace("_", " ").replace("-", " ")
        words = [w for w in query_clean.split() if len(w) > 2]
        
        scores = []
        for doc in self.documents:
            doc_text = doc["text"].lower()
            score = 0
            
            # Match against disease name metadata
            meta_name = doc.get("metadata", {}).get("disease_name", "").lower()
            if meta_name:
                meta_clean = meta_name.replace("_", " ").replace("-", " ")
                if meta_clean in query_clean or query_clean in meta_clean:
                    score += 100
                    
            # Check overlap of other words
            for word in words:
                if word in doc_text:
                    score += 10
            scores.append(score)
            
        top_indices = np.argsort(scores)[::-1][:top_k]
        context_blocks = []
        for idx in top_indices:
            context_blocks.append(self.documents[idx]["text"])
            
        return "\n---\n".join(context_blocks)

plant_rag = PlantRAG()
