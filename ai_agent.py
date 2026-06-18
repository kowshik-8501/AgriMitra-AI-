import os
import json
# pyrefly: ignore [missing-import]
import httpx
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

import pandas as pd

PLACEHOLDER_KEYS = {
    "YOUR_GEMINI_API_KEY_HERE",
}

class PlantAIAgent:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        
        # Check if keys are placeholders or valid
        self._is_valid_gemini = bool(self.gemini_key and self.gemini_key not in PLACEHOLDER_KEYS and not self.gemini_key.startswith("AQ.Ab8RN6LWV"))
        self._is_valid_openai = bool(self.openai_key and self.openai_key.startswith("sk-"))
        self._is_valid_key = self._is_valid_gemini or self._is_valid_openai
        
        self.base_dir = os.path.dirname(__file__)
        self.disease_csv = os.path.join(self.base_dir, "disease_info.csv")
        self.supplement_csv = os.path.join(self.base_dir, "supplement_info.csv")
        try:
            self.disease_df = pd.read_csv(self.disease_csv, encoding='cp1252')
            self.supplement_df = pd.read_csv(self.supplement_csv, encoding='cp1252')
        except Exception as e:
            print(f"Error loading CSVs in PlantAIAgent: {e}")
            self.disease_df = None
            self.supplement_df = None
        
        if not self._is_valid_key:
            print("WARNING: Neither GEMINI_API_KEY nor OPENAI_API_KEY is set or is valid. PlantAIAgent running in fallback mode.")

    def generate_report(self, disease_name: str, rag_context: str, weather_data: dict, risk_data: dict, lang: str = "en") -> dict:
        """
        Runs multi-step prompt engineering to generate a comprehensive 11-field agricultural report.
        Forces JSON output format via raw REST API. Supports both Gemini and OpenAI backend models dynamically.
        """
        if not self._is_valid_key:
            return self.get_fallback_report(disease_name, lang)

        lang_names = {
            "en": "English", "te": "Telugu", "ta": "Tamil",
            "ml": "Malayalam", "hi": "Hindi", "kn": "Kannada", "or": "Odia"
        }
        target_lang = lang_names.get(lang, "English")

        prompt = f"""
You are an expert plant pathologist and agricultural advisor. Your job is to generate a comprehensive, highly detailed agricultural report for a farmer regarding a diagnosed plant disease.

---
DIAGNOSED DISEASE: {disease_name}
CONFIDENCE LEVEL: {risk_data.get('confidence', 90.0)}%
RECOVERY PROBABILITY: {risk_data.get('recovery_prob', 80.0)}%
ENVIRONMENTAL WEATHER DETAILS:
- Temperature: {weather_data.get('temp')}°C
- Relative Humidity: {weather_data.get('humidity')}%
- Current Rain: {weather_data.get('rain')} mm
- Soil Moisture State: {weather_data.get('soil_moisture')}
- Local Climate Spread Risk: {risk_data.get('spread_risk')} (Score: {risk_data.get('spread_risk_score')}/100)

RAG KNOWLEDGE BASE CONTEXT (Verifiable facts from database):
{rag_context}
---

Generate a detailed report structured exactly in the following JSON format.
Ensure all text values in the JSON output are translated and written entirely in: {target_lang}.

JSON SCHEMA:
{{
  "description": "A detailed explanation of the disease, what it is, and its general impact on this plant.",
  "symptoms": ["A list of specific visual patterns, spots, leaf shapes, color changes, or structural signs of infection on the plant"],
  "causes": ["List the biological factors, pathogens, weather triggers, or poor farming practices that caused the outbreak"],
  "weather_impact": "Assess how the current weather conditions (temperature, humidity, rain) are impacting the disease. State whether it encourages spread or suppresses it and why.",
  "treatment_organic": "Step-by-step organic/biological control measures, home remedies, and soil amendments.",
  "treatment_chemical": "Step-by-step chemical control measures, specific active ingredients (e.g. fungicides, copper sprays) with exact application instructions and safety warnings.",
  "fertilizer_pesticide_suggestions": "Nutrient management advice. Suggest specific fertilizers (NPK adjustment) or soil nutrients to help the plant recover, and how to apply them.",
  "action_plan_7_day": {{
    "Day 1-2": "Immediate steps to isolate or prune infected parts, prepare sprays.",
    "Day 3-4": "Application of organic/chemical treatments and soil nutrient amendments.",
    "Day 5-6": "Monitoring spread, irrigation adjustment, and vector insect control.",
    "Day 7": "Final check, cleanup, and preventive spraying for neighboring plants."
  }},
  "recovery_explanation": "Explain the reasoning behind the {risk_data.get('recovery_prob')}% recovery probability. Detail what factors (treatment speed, weather, plant health) will improve or degrade this likelihood."
}}
"""
        # 1. Try Gemini first if valid
        if self._is_valid_gemini:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_key}"
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "responseMimeType": "application/json"
                }
            }
            try:
                with httpx.Client(timeout=60.0) as client:
                    res = client.post(url, json=payload)
                    if res.status_code == 200:
                        response_json = res.json()
                        generated_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
                        return json.loads(generated_text)
                    else:
                        print(f"Gemini REST Error {res.status_code}: {res.text}")
            except Exception as e:
                print(f"Error generating report from Gemini REST API: {e}")

        # 2. Try OpenAI as fallback if valid
        if self._is_valid_openai:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "response_format": { "type": "json_object" }
            }
            try:
                with httpx.Client(timeout=30.0) as client:
                    res = client.post(url, headers=headers, json=payload)
                    if res.status_code == 200:
                        response_json = res.json()
                        generated_text = response_json["choices"][0]["message"]["content"]
                        return json.loads(generated_text)
                    else:
                        print(f"OpenAI REST Error {res.status_code}: {res.text}")
            except Exception as e:
                print(f"Error generating report from OpenAI API: {e}")

        return self.get_fallback_report(disease_name, lang)

    def chat_response(self, user_message: str, chat_history: list, disease_name: str, report_data: dict, weather_data: dict, lang: str = "en") -> str:
        """
        Maintains conversational Q&A with context grounding.
        Falls back to intelligent offline answers when API key is missing.
        """
        # Intercept common greetings
        msg_lower = user_message.lower().strip().rstrip('?!.')
        greetings = ["hi", "hello", "hey", "hii", "hola", "greetings", "good morning", "good afternoon", "good evening"]
        if msg_lower in greetings or any(msg_lower.startswith(g + " ") for g in greetings):
            lang_greetings = {
                "en": "Hello! How can I assist you today? I'm here to help you manage and treat your plant's disease.",
                "hi": "नमस्ते! मैं आज आपकी क्या सहायता कर सकता हूँ? मैं आपके पौधे की बीमारी के इलाज में मदद के लिए यहाँ हूँ।",
                "te": "నమస్తే! ఈరోజు నేను మీకు ఎలా సహాయపడగలను? మీ మొక్క తెగులు నివారణలో సహాయం చేయడానికి నేను ఇక్కడ ఉన్నాను.",
                "ta": "வணக்கம்! இன்று நான் உங்களுக்கு எவ்வாறு உதவ முடியும்? உங்கள் தாவர நோயைக் குணப்படுத்த உதவ நான் இங்கு இருக்கிறேன்.",
                "ml": "ஹலோ! ഞാൻ ഇന്ന് നിങ്ങളെ എങ്ങനെ സഹായിക്കണം? നിങ്ങളുടെ ചെടിയുടെ രോഗം മാറ്റാൻ സഹായിക്കാൻ ഞാൻ ഇവിടെയുണ്ട്.",
                "kn": "ನಮಸ್ತೆ! ಇಂದು ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ? ನಿಮ್ಮ ಸಸ್ಯದ ರೋಗವನ್ನು ನಿವಾರಿಸಲು ಸಹಾಯ ಮಾಡಲು ನಾನು ಇಲ್ಲಿದ್ದೇನೆ.",
                "or": "ନମସ୍କାର! ଆଜି ମୁଁ ଆପଣଙ୍କୁ କିପରି ସାହାଯ୍ୟ କରିପାରିବି? ଆପଣଙ୍କ ଗଛର ରୋଗ ନିବାରଣ ପାଇଁ ମୁଁ ଏଠାରେ ଅଛି।"
            }
            return lang_greetings.get(lang, lang_greetings["en"])

        if not self._is_valid_key:
            return self._offline_chat(user_message, disease_name, report_data, weather_data, lang)

        lang_names = {
            "en": "English", "te": "Telugu", "ta": "Tamil",
            "ml": "Malayalam", "hi": "Hindi", "kn": "Kannada", "or": "Odia"
        }
        target_lang = lang_names.get(lang, "English")

        formatted_history = []
        for msg in chat_history[-6:]:  # Keep last 3 turns
            formatted_history.append(f"{msg['role'].upper()}: {msg['content']}")
            
        history_str = "\n".join(formatted_history)

        # Search FAISS vector store for dynamically related details to answer user's free-form chat queries
        from rag import plant_rag
        dynamic_context = plant_rag.search_vector_index(user_message, top_k=1)

        prompt = f"""
You are "Plant Doctor AI", a helpful, friendly agricultural assistant.
You are helping a farmer whose plant is diagnosed with: {disease_name}.
Current weather details: {weather_data.get('temp')}°C, {weather_data.get('humidity')}% Humidity.

Here is the diagnosis report we generated for the farmer:
{json.dumps(report_data, indent=2)}

DYNAMIC ADDITIONAL CONTEXT RETRIEVED FROM KNOWLEDGE BASE (via FAISS Vector Search):
{dynamic_context}

Guidelines:
1. Ground your answers strictly on the diagnosis report, the retrieved dynamic context, RAG database rules, and best agricultural practices.
2. If the user asks about other diseases, treatments, or crops mentioned in the retrieved dynamic context, use that information to help them.
3. Be polite, clear, and action-oriented. Keep paragraphs short and use bullet points where helpful.
4. You must respond in the target language: {target_lang}.
5. If the user asks general questions, guide them back to caring for their plant.
6. Use the Google Search tool when necessary to find the most accurate and up-to-date treatment or preventative steps for the crop's condition.

CONVERSATION HISTORY:
{history_str}
USER: {user_message}
ASSISTANT:"""

        # 1. Try Gemini first if valid
        if self._is_valid_gemini:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_key}"
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "tools": [
                    {
                        "google_search": {}
                    }
                ]
            }
            try:
                with httpx.Client(timeout=30.0) as client:
                    res = client.post(url, json=payload)
                    if res.status_code == 200:
                        response_json = res.json()
                        return response_json["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        print(f"Chat API Error {res.status_code}: {res.text}")
            except Exception as e:
                print(f"Error in REST chat agent: {e}")

        # 2. Try OpenAI as fallback if valid
        if self._is_valid_openai:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            try:
                with httpx.Client(timeout=15.0) as client:
                    res = client.post(url, headers=headers, json=payload)
                    if res.status_code == 200:
                        response_json = res.json()
                        return response_json["choices"][0]["message"]["content"]
                    else:
                        print(f"OpenAI Chat API Error {res.status_code}: {res.text}")
            except Exception as e:
                print(f"Error in OpenAI chat agent: {e}")

        # If all API calls failed, try offline response instead of a generic error
        return self._offline_chat(user_message, disease_name, report_data, weather_data, lang)

    def _offline_chat(self, user_message: str, disease_name: str, report_data: dict, weather_data: dict, lang: str = "en") -> str:
        """
        Provides intelligent offline chat responses using the report data.
        Keyword matching maps user questions to relevant report sections.
        """
        msg_lower = user_message.lower()
        
        # Keyword-based routing into report sections
        if any(kw in msg_lower for kw in ["organic", "natural", "home remedy", "eco", "neem", "biological"]):
            answer = report_data.get("treatment_organic", "")
            if answer:
                return f"🌿 Organic Treatment for {disease_name}:\n\n{answer}"
        
        if any(kw in msg_lower for kw in ["chemical", "fungicide", "pesticide", "spray", "copper", "dosage"]):
            answer = report_data.get("treatment_chemical", "")
            if answer:
                return f"🧪 Chemical Treatment for {disease_name}:\n\n{answer}"
        
        if any(kw in msg_lower for kw in ["fertilizer", "nutrient", "npk", "nitrogen", "potassium", "phosphorus", "soil"]):
            answer = report_data.get("fertilizer_pesticide_suggestions", "")
            if answer:
                return f"🌱 Fertilizer & Nutrient Advice:\n\n{answer}"
        
        if any(kw in msg_lower for kw in ["symptom", "sign", "look like", "identify", "spot", "leaf"]):
            symptoms = report_data.get("symptoms", [])
            if symptoms:
                sym_text = "\n".join([f"• {s}" for s in symptoms])
                return f"🔍 Key Symptoms of {disease_name}:\n\n{sym_text}"
        
        if any(kw in msg_lower for kw in ["cause", "why", "reason", "how did", "origin"]):
            causes = report_data.get("causes", [])
            if causes:
                cause_text = "\n".join([f"• {c}" for c in causes])
                return f"⚠️ Root Causes of {disease_name}:\n\n{cause_text}"
        
        if any(kw in msg_lower for kw in ["weather", "climate", "rain", "humidity", "temperature", "spread", "contagious"]):
            answer = report_data.get("weather_impact", "")
            if answer:
                temp = weather_data.get('temp', 'N/A')
                hum = weather_data.get('humidity', 'N/A')
                return f"🌦️ Weather Impact Analysis (Current: {temp}°C, {hum}% humidity):\n\n{answer}"
        
        if any(kw in msg_lower for kw in ["plan", "schedule", "day", "timeline", "step", "what to do"]):
            plan = report_data.get("action_plan_7_day", {})
            if plan:
                plan_text = "\n".join([f"📅 {k}: {v}" for k, v in plan.items()])
                return f"📋 7-Day Action Plan for {disease_name}:\n\n{plan_text}"
        
        if any(kw in msg_lower for kw in ["recover", "cure", "save", "heal", "chance", "probability"]):
            answer = report_data.get("recovery_explanation", "")
            if answer:
                return f"💪 Recovery Outlook for {disease_name}:\n\n{answer}"
        
        # Offline search fallback: perform a local keyword search to find matches for other diseases
        from rag import plant_rag
        dynamic_context = plant_rag.fallback_text_search(user_message, top_k=1)
        
        # Default: provide a summary from the description or dynamic search results
        desc = report_data.get("description", "")
        if dynamic_context and "RAG database is empty" not in dynamic_context and len(dynamic_context) > 20:
            return (
                f"🌾 Offline Knowledge Base Search Result:\n\n{dynamic_context}\n\n"
                f"ℹ️ Note: For dynamic AI-powered responses, please check that GEMINI_API_KEY is valid and your daily/RPM free tier API limits are not exhausted."
            )
            
        return (
            f"🌾 About {disease_name}:\n\n{desc}\n\n"
            f"💡 You can ask me about:\n"
            f"• Organic treatments\n"
            f"• Chemical sprays & dosages\n"
            f"• Fertilizer recommendations\n"
            f"• Symptoms & causes\n"
            f"• Weather impact on spread\n"
            f"• 7-day action plan\n"
            f"• Recovery chances\n\n"
            f"ℹ️ Note: For dynamic AI-powered responses, please check that GEMINI_API_KEY is valid and your daily/RPM free tier API limits are not exhausted."
        )

    def get_fallback_report(self, disease_name: str, lang: str = "en") -> dict:
        """
        Offline fallback data that pulls actual facts from local CSVs if Gemini API Key is missing.
        """
        desc = f"Offline Report for {disease_name}. To get a detailed, dynamic AI report, check that GEMINI_API_KEY is valid and your free tier API limits are not exhausted."
        prevent = "Isolate the plant, remove heavily infected foliage, and apply suitable treatment."
        supp_name = "N/A"
        buy_link = ""
        
        if self.disease_df is not None:
            # Normalize to match disease name strings
            normalized_query = disease_name.lower().replace(" ", "").replace("-", "").replace("_", "")
            match_row = None
            match_idx = None
            for idx, row in self.disease_df.iterrows():
                db_name = str(row.get('disease_name', '')).lower().replace(" ", "").replace("-", "").replace("_", "")
                if normalized_query in db_name or db_name in normalized_query:
                    match_row = row
                    match_idx = idx
                    break
            
            if match_row is not None:
                desc = match_row.get('description', desc)
                prevent = match_row.get('Possible Steps', prevent)
                if self.supplement_df is not None and match_idx is not None and match_idx < len(self.supplement_df):
                    supp_name = self.supplement_df.iloc[match_idx].get('supplement name', 'N/A')
                    buy_link = self.supplement_df.iloc[match_idx].get('buy link', '')

        return {
            "description": desc,
            "symptoms": [
                "Leaves showing visible lesions, spots, or chlorosis.",
                f"Typical visual symptoms of {disease_name}."
            ],
            "causes": [
                "Environmental pathogen transmission",
                "High relative humidity and leaf moisture levels"
            ],
            "weather_impact": "Offline mode: cannot evaluate weather conditions. Check API Key.",
            "treatment_organic": f"Organic/Cultural recommendations:\n{prevent}\n\nApply neem oil or other organic fungicides to strengthen plant defenses.",
            "treatment_chemical": f"Chemical recommendations:\nUse a suitable broad-spectrum fungicide or bactericide targeting {disease_name}. Follow manufacturer guidelines for safety and application rates.",
            "fertilizer_pesticide_suggestions": f"Use {supp_name} to address nutrient deficiencies. Purchase here: {buy_link}" if buy_link else f"Adjust fertilizer balance to enhance potassium and silicon levels for cell wall strength.",
            "action_plan_7_day": {
                "Day 1-2": "Isolate the plant, prune and destroy infected foliage.",
                "Day 3-4": f"Apply organic neem oil spray or target fungicide treatment for {disease_name}.",
                "Day 5-6": "Optimize irrigation (avoid wet foliage, irrigate root zone).",
                "Day 7": "Inspect for new symptoms and repeat treatment if spread persists."
            },
            "recovery_explanation": "Recovery depends on how early the disease is detected. If caught early and infected parts are removed, recovery likelihood is high."
        }

# Export agent singleton instance
plant_agent = PlantAIAgent()
