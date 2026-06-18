import os
import uuid
import time
import json
import pickle
import requests
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from pathlib import Path
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]
from fastapi.templating import Jinja2Templates
# pyrefly: ignore [missing-import]
from gtts import gTTS
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torchvision.transforms.functional as TF
# pyrefly: ignore [missing-import]
import joblib

# Load environment variables
load_dotenv(Path(__file__).with_name('.env'))

# Import Deep Analysis App components
from predictor import predict_disease
from rag import plant_rag
from risk_model import fetch_weather, calculate_risk_and_recovery
from ai_agent import plant_agent

# Directories
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
AUDIO_FOLDER = BASE_DIR / 'static' / 'audio'
for folder in [UPLOAD_FOLDER, AUDIO_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# Translation Cache
_translation_cache = {}

# Utility functions (same as Flask version)
def translate_text(text: str, target_lang: str) -> str:
    if not text or target_lang == 'en':
        return text
    cache_key = (text, target_lang)
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]
    lang_map = {
        'hi': 'hi', 'te': 'te', 'ta': 'ta', 'kn': 'kn',
        'ml': 'ml', 'or': 'or'
    }
    target = lang_map.get(target_lang, 'hi')
    try:
        from deep_translator import GoogleTranslator
        res = GoogleTranslator(source='auto', target=target).translate(text)
        _translation_cache[cache_key] = res
        return res
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def sanitize_text(text: str) -> str:
    if not text:
        return ""
    import re
    text = re.sub(r'<[^>]*>', '', text)
    return " ".join(text.split())

def cleanup_old_audio(folder: Path, max_age_seconds: int = 3600):
    now = time.time()
    for f in folder.iterdir():
        if f.is_file() and f.suffix == '.mp3' and now - f.stat().st_mtime > max_age_seconds:
            f.unlink()

def create_audio(text: str, lang_code: str) -> str | None:
    try:
        cleanup_old_audio(AUDIO_FOLDER)
        clean_text = sanitize_text(text)
        if not clean_text:
            return None
        lang_map = {
            'hi': 'hi', 'te': 'te', 'ta': 'ta', 'kn': 'kn',
            'en': 'en', 'ml': 'ml', 'or': 'or'
        }
        target = lang_map.get(lang_code, 'en')
        tts = gTTS(text=clean_text, lang=target)
        filename = f"speech_{uuid.uuid4().hex}.mp3"
        filepath = AUDIO_FOLDER / filename
        tts.save(str(filepath))
        return filename
    except Exception as e:
        print(f"Audio creation error: {e}")
        return None



# Load CSV data
try:
    disease_info = pd.read_csv(BASE_DIR / "disease_info.csv", encoding='cp1252')
    supplement_info = pd.read_csv(BASE_DIR / "supplement_info.csv", encoding='cp1252')
except Exception as e:
    print(f"CSV load error: {e}")
    disease_info = pd.DataFrame()
    supplement_info = pd.DataFrame()

# Load crop recommendation model
crop_model = None
try:
    with open(BASE_DIR / 'crop_model.pkl', 'rb') as f:
        crop_model = pickle.load(f)
except Exception as e:
    print(f"Crop model load error: {e}")

# Helper data for crops
CROP_DETAILS = {
    'rice': "Rice is a staple cereal grain, primarily grown in warm, wet climates. It is rich in carbohydrates and requires significant water for cultivation.",
    'maize': "Maize (corn) is a versatile cereal crop adapted to a wide range of climates. It grows best in well-drained, fertile soils with moderate rainfall.",
    'chickpea': "Chickpea is an important pulse crop that thrives in cool, dry conditions. It is highly nutritious, providing proteins and essential minerals.",
    'kidneybeans': "Kidney beans grow well in warm climates and require moist, fertile soil. They are an excellent vegetarian source of protein.",
    'pigeonpeas': "Pigeonpea is a drought-tolerant legume crop grown in tropical and subtropical regions. It helps in nitrogen fixation in soil.",
    'mothbeans': "Moth beans are highly drought-resistant legumes grown in arid regions. They are a valuable crop for soil conservation.",
    'mungbean': "Mungbeans are short-duration warm-season legumes. They require warm temperatures and moderate rainfall.",
    'blackgram': "Blackgram (Urad dal) is a tropical legume crop. It thrives in warm temperatures and requires well-drained loamy soils.",
    'lentil': "Lentils are cool-season food legumes. They require cool temperatures and can grow on wide variety of soils.",
    'pomegranate': "Pomegranate is a popular fruit crop grown in warm, dry climates. It is rich in antioxidants and prefers well-drained soils.",
    'banana': "Bananas are tropical fruit crops that require warm, humid climates, rich organic soil, and abundant water.",
    'mango': "Mango is a tropical fruit tree. It thrives in warm climates with distinct dry seasons for flowering and fruiting.",
    'grapes': "Grapes are grown in temperate and warm climates. They require well-drained soil, sunny conditions, and support structures (trellises).",
    'watermelon': "Watermelons thrive in warm temperatures with long, hot growing seasons. They require sandy, well-drained soils.",
    'muskmelon': "Muskmelons prefer warm temperatures and well-drained, sandy loam soils. They require moderate watering.",
    'apple': "Apples are temperate fruit crops that require winter chilling for proper bud development and fertile, well-drained soils.",
    'orange': "Oranges are subtropical citrus crops. They require warm temperatures, frost-free climates, and regular irrigation.",
    'papaya': "Papayas grow best in tropical, frost-free climates with fertile, well-drained soils and regular water supply.",
    'coconut': "Coconuts are coastal tropical palms. They require warm, humid climates, sandy soil, and high rainfall.",
    'cotton': "Cotton is a major fiber crop. It requires warm temperatures, moderate rainfall, and fertile, deep black soils.",
    'jute': "Jute is a natural fiber crop that grows best in warm, humid climates with high rainfall and alluvial soils.",
    'coffee': "Coffee is a tropical crop grown in shade. It requires cool to warm temperatures, high rainfall, and rich, well-drained soils."
}
CROP_CENTERS = {
    'rice': [80, 47, 40, 23, 82, 6.4, 236],
    'maize': [77, 48, 19, 22, 65, 6.2, 84],
    'chickpea': [40, 67, 79, 18, 16, 7.3, 80],
    'kidneybeans': [20, 67, 20, 20, 21, 5.7, 105],
    'pigeonpeas': [20, 67, 20, 27, 48, 5.7, 149],
    'mothbeans': [21, 48, 20, 28, 53, 6.8, 51],
    'mungbean': [20, 48, 20, 28, 85, 6.7, 48],
    'blackgram': [40, 67, 19, 29, 65, 7.2, 67],
    'lentil': [18, 68, 19, 18, 64, 6.9, 45],
    'pomegranate': [40, 18, 40, 21, 88, 6.4, 107],
    'banana': [100, 82, 50, 27, 80, 5.9, 104],
    'mango': [20, 27, 30, 31, 50, 5.7, 94],
    'grapes': [23, 132, 200, 23, 81, 6.0, 69],
    'watermelon': [99, 17, 50, 25, 85, 6.4, 50],
    'muskmelon': [100, 17, 50, 28, 92, 6.3, 24],
    'apple': [20, 137, 199, 22, 92, 5.9, 112],
    'orange': [39, 16, 10, 22, 92, 7.0, 110],
    'papaya': [49, 59, 50, 33, 92, 6.7, 142],
    'coconut': [21, 15, 30, 27, 96, 5.9, 175],
    'cotton': [117, 46, 19, 23, 79, 6.9, 80],
    'jute': [78, 46, 39, 24, 79, 6.7, 174],
    'coffee': [101, 28, 29, 25, 58, 6.7, 158]
}

def get_crop_reason(crop, n, p, k, temp, hum, ph, rain):
    crop = crop.lower()
    if crop not in CROP_CENTERS:
        return "This crop is well-suited for the combination of nutrients and climate in your area."
    center = CROP_CENTERS[crop]
    reasons = []
    if abs(n - center[0]) < 20: reasons.append(f"the Nitrogen level ({n}) is ideal")
    if abs(p - center[1]) < 20: reasons.append(f"Phosphorus ({p}) is in the optimal range")
    if abs(k - center[2]) < 20: reasons.append(f"Potassium ({k}) supports its growth")
    if abs(temp - center[3]) < 5: reasons.append(f"the temperature ({temp}°C) is perfect")
    if abs(hum - center[4]) < 10: reasons.append(f"humidity ({hum}%) matches its needs")
    if abs(ph - center[5]) < 0.5: reasons.append(f"soil pH ({ph}) is just right")
    if abs(rain - center[6]) < 30: reasons.append(f"rainfall ({rain}mm) provides the necessary moisture")
    if not reasons:
        return "This crop is the best match for your overall soil profile and weather conditions."
    return f"{crop.capitalize()} is recommended because " + ", ".join(reasons[:3]) + "."

# Fertilizer advisor (same as Flask)
class FertilizerAdvisor:
    def recommend(self, n, p, k, crop):
        targets = {
            'Paddy': (100, 50, 50), 'Maize': (120, 60, 40), 'Wheat': (120, 60, 40),
            'Sugarcane': (150, 80, 80), 'Cotton': (100, 50, 50), 'Rice': (100, 50, 50)
        }
        tn, tp, tk = targets.get(crop, (100, 50, 50))
        recs = []
        if n < tn - 15: recs.append("Apply Urea to increase Nitrogen levels.")
        if p < tp - 10: recs.append("Apply DAP (Di-Ammonium Phosphate) for Phosphorus.")
        if k < tk - 10: recs.append("Apply MOP (Muriate of Potash) for Potassium.")
        if not recs:
            recs.append("Your soil NPK levels are optimal for this crop. Maintain organic matter.")
        return recs

fertilizer_advisor = FertilizerAdvisor()

# Yield predictor (same as Flask)
class YieldPredictor:
    def __init__(self):
        self.model_path = BASE_DIR / 'models' / 'yield_model.pkl'
        self.le_dist_path = BASE_DIR / 'models' / 'le_district.pkl'
        self.le_crop_path = BASE_DIR / 'models' / 'le_crop.pkl'
        self.data_dir = BASE_DIR / 'data'
        self.load_artifacts()
    def load_artifacts(self):
        try:
            self.model = joblib.load(self.model_path)
            self.le_district = joblib.load(self.le_dist_path)
            self.le_crop = joblib.load(self.le_crop_path)
            self.history = pd.read_csv(self.data_dir / 'historical_yield.csv')
        except Exception as e:
            print(f"Error loading yield artifacts: {e}")
            self.model = None
    def predict(self, input_data):
        try:
            dist_enc = self.le_district.transform([input_data.get('district', 'Guntur')])[0]
            crop_enc = self.le_crop.transform([input_data.get('crop', 'Rice')])[0]
            rain = float(input_data.get('rainfall', 1000))
            temp = float(input_data.get('temp', 28))
            features = np.array([[dist_enc, crop_enc, rain, temp,
                                 float(input_data.get('n', 80)),
                                 float(input_data.get('p', 40)),
                                 float(input_data.get('k', 40))]])
            if self.model:
                pred = self.model.predict(features)[0]
            else:
                pred = 3.5
            base_val = float(self.history[self.history['District'].str.contains(input_data.get('district', ''), case=False)]['Yield'].mean() or pred)
            trend = [float(base_val * (1 + np.random.uniform(-0.1, 0.1))) for _ in range(5)]
            trend.append(float(round(pred, 2)))
            return float(round(pred, 2)), trend
        except Exception as e:
            print(f"Yield prediction error: {e}")
            return 3.5, [3.0, 3.2, 3.1, 3.4, 3.3, 3.5]

yield_predictor = YieldPredictor()

# FastAPI app
app = FastAPI(title="AgriMitra AI 🌾")
app.mount('/static', StaticFiles(directory=str(BASE_DIR / 'static'), html=True), name='static')

from jinja2 import pass_context

class CustomJinja2Templates(Jinja2Templates):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Map Flask-style route names to FastAPI function names
        _flask_to_fastapi = {
            'home_page': 'home',
            'ai_engine_page': 'ai_engine',
            'yield_prediction': 'yield_page',
            'fertilizer_recommendation': 'fertilizer_page',
            'contact_page': 'contact',
            'privacy_page': 'privacy',
        }
        @pass_context
        def custom_url_for(context: dict, name: str, /, **path_params: any):
            request = context["request"]
            # Translate Flask route name to FastAPI function name if needed
            name = _flask_to_fastapi.get(name, name)
            if name == "static" and "filename" in path_params:
                path_params["path"] = path_params.pop("filename")
            url = request.url_for(name, **path_params)
            return url.path + (f"?{url.query}" if url.query else "")
        self.env.globals["url_for"] = custom_url_for

templates = CustomJinja2Templates(directory=[
    str(BASE_DIR / 'templates')
])

# Home page
@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name='home.html')

@app.get('/index', response_class=HTMLResponse)
async def ai_engine(request: Request):
    return templates.TemplateResponse(request=request, name='index.html')

# Image submission route (POST)
@app.post('/submit', response_class=HTMLResponse)
async def submit(request: Request, image: UploadFile = File(...), language: str = Form('en')):
    try:
        filename = f"{uuid.uuid4().hex}_{image.filename}"
        file_path = UPLOAD_FOLDER / filename
        with open(file_path, 'wb') as f:
            f.write(await image.read())
        pred_idx, _, _ = predict_disease(str(file_path))
        pred_idx = int(pred_idx)
        if pred_idx >= len(disease_info):
            pred_idx = 0
        title = disease_info['disease_name'][pred_idx]
        description = disease_info['description'][pred_idx]
        prevent = disease_info['Possible Steps'][pred_idx]
        image_url = disease_info['image_url'][pred_idx]
        supplement_name = supplement_info['supplement name'][pred_idx]
        supplement_image_url = supplement_info['supplement image'][pred_idx]
        supplement_buy_link = supplement_info['buy link'][pred_idx]
        # Translations
        lang_note = translate_text(title, language) if language != 'en' else None
        display_desc = translate_text(description, language) if language != 'en' else description
        display_prevent = translate_text(prevent, language) if language != 'en' else prevent
        audio_text = f"Diagnosis result: {lang_note if lang_note else title}. Details: {display_desc}. Prevention: {display_prevent}."
        audio_file = None
        return templates.TemplateResponse(request=request, name='submit.html', context={
            'title': title,
            'desc': display_desc,
            'prevent': display_prevent,
            'image_url': image_url,
            'pred': pred_idx,
            'sname': supplement_name,
            'simage': supplement_image_url,
            'buy_link': supplement_buy_link,
            'selected_lang': language,
            'lang_note': lang_note,
            'audio_text': audio_text,
            'audio_file': audio_file,
            'filename': filename
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during submission: {str(e)}")

# Market page
@app.get('/market', response_class=HTMLResponse)
async def market(request: Request):
    return templates.TemplateResponse(request=request, name='market.html', context={
        'supplement_image': list(supplement_info['supplement image']),
        'supplement_name': list(supplement_info['supplement name']),
        'disease': list(disease_info['disease_name']),
        'buy': list(supplement_info['buy link'])
    })

@app.get('/contact', response_class=HTMLResponse)
async def contact(request: Request):
    return templates.TemplateResponse(request=request, name='contact-us.html')

@app.get('/privacy', response_class=HTMLResponse)
async def privacy(request: Request):
    return templates.TemplateResponse(request=request, name='privacy.html')

@app.get('/documentation')
async def documentation():
    return FileResponse(BASE_DIR / 'LICENSE', media_type='text/plain')

# Crop recommendation
@app.get('/crop', response_class=HTMLResponse)
async def crop_page(request: Request):
    return templates.TemplateResponse(request=request, name='crop.html')

@app.post('/crop', response_class=HTMLResponse)
async def crop_predict(request: Request,
                      nitrogen: float = Form(0), phosphorus: float = Form(0), potassium: float = Form(0),
                      temperature: float = Form(0), humidity: float = Form(0), ph: float = Form(0), rainfall: float = Form(0),
                      language: str = Form('en')):
    try:
        input_features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        pred = crop_model.predict(input_features)[0]
        crop_name = str(pred).capitalize()
        raw_desc = CROP_DETAILS.get(pred.lower(), "Information not available.")
        raw_reason = get_crop_reason(pred, nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
        if language != 'en':
            language_note = translate_text(crop_name, language)
            crop_desc = translate_text(raw_desc, language)
            crop_reason = translate_text(raw_reason, language)
        else:
            language_note = None
            crop_desc = raw_desc
            crop_reason = raw_reason
        audio_text = f"The AI recommends planting {language_note if language_note else crop_name}. {crop_desc} {crop_reason}"
        audio_file = None
        return templates.TemplateResponse(request=request, name='crop.html', context={
            'crop': crop_name,
            'crop_desc': crop_desc,
            'crop_reason': crop_reason,
            'language_note': language_note,
            'audio_text': audio_text,
            'selected_lang': language,
            'audio_file': audio_file
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Yield prediction
@app.get('/yield', response_class=HTMLResponse)
async def yield_page(request: Request):
    return templates.TemplateResponse(request=request, name='yield.html', context={
        'input_data': {'district': '', 'crop': 'Rice', 'n': 80, 'p': 40, 'k': 40, 'temp': 28, 'rainfall': 1200}
    })

@app.post('/yield', response_class=HTMLResponse)
async def yield_predict(request: Request,
                       district: str = Form(''), crop: str = Form(''),
                       n: float = Form(0), p: float = Form(0), k: float = Form(0),
                       temp: float = Form(0), rainfall: float = Form(0),
                       language: str = Form('en')):
    try:
        input_data = {'district': district.strip(), 'crop': crop, 'n': n, 'p': p, 'k': k, 'temp': temp, 'rainfall': rainfall}
        pred, trend = yield_predictor.predict(input_data)
        # Regional comparison
        hist = yield_predictor.history
        crop_matches = hist[hist['Crop'].str.contains(crop, case=False)]
        if crop_matches.empty:
            top_regions = hist.groupby('District')['Yield'].mean().sort_values(ascending=False).head(5)
        else:
            top_regions = crop_matches.groupby('District')['Yield'].mean().sort_values(ascending=False).head(5)
        regional_data = {
            'labels': list(top_regions.index),
            'chart_values': [round(float(v), 2) for v in top_regions.values],
            'current_district': district,
            'current_val': pred
        }
        # Status text
        if pred > 4.5:
            status = "Excellent"
            advice = "Conditions are perfect. Consider early market booking."
        elif pred > 3.0:
            status = "Good"
            advice = "Standard maintenance will secure this harvest."
        else:
            status = "Average"
            advice = "Review soil nutrients and irrigation frequency."
        report_text = f"The predicted yield for {crop} in {district} is {pred} tons per hectare. Status: {status}. {advice}"
        trans_result = translate_text(report_text, language) if language != 'en' else report_text
        audio_file = None
        return templates.TemplateResponse(request=request, name='yield.html', context={
            'result': pred,
            'input_data': input_data,
            'trend': trend,
            'regional_data': regional_data,
            'lang': language,
            'trans_result': trans_result,
            'audio_text': trans_result,
            'selected_lang': language,
            'audio_file': audio_file
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Fertilizer recommendation
@app.get('/fertilizer', response_class=HTMLResponse)
async def fertilizer_page(request: Request):
    return templates.TemplateResponse(request=request, name='fertilizer.html', context={'crops': ['Paddy', 'Maize', 'Sugarcane', 'Cotton', 'Wheat', 'Rice']})

@app.post('/fertilizer', response_class=HTMLResponse)
async def fertilizer_recommend(request: Request,
                              nitrogen: float = Form(0), phosphorus: float = Form(0), potassium: float = Form(0),
                              crop: str = Form('Paddy'), language: str = Form('en')):
    try:
        recs = fertilizer_advisor.recommend(nitrogen, phosphorus, potassium, crop)
        if language != 'en':
            trans_recs = [translate_text(r, language) for r in recs]
            audio_text = '. '.join(trans_recs)
        else:
            trans_recs = recs
            audio_text = '. '.join(recs)
        audio_file = None
        return templates.TemplateResponse(request=request, name='fertilizer.html', context={
            'recs': recs,
            'trans_recs': trans_recs,
            'data': {'n': nitrogen, 'p': phosphorus, 'k': potassium, 'crop': crop},
            'selected_lang': language,
            'audio_text': audio_text,
            'audio_file': audio_file,
            'crops': ['Paddy', 'Maize', 'Sugarcane', 'Cotton', 'Wheat', 'Rice']
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Explore diseases
@app.get('/explore', response_class=HTMLResponse)
async def explore(request: Request):
    diseases = disease_info.to_dict(orient='records') if not disease_info.empty else []
    return templates.TemplateResponse(request=request, name='explore.html', context={'diseases': diseases})

# API endpoints (same as Flask)
@app.post('/api/detect-location')
async def detect_location(data: dict):
    try:
        lat = data.get('lat')
        lon = data.get('lon')
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10"
        headers = {'User-Agent': 'AgriMitraAI/1.0'}
        resp = requests.get(url, headers=headers).json()
        addr = resp.get('address', {})
        state = addr.get('state', '').lower()
        mapping = {
            'andhra pradesh': 'te', 'telangana': 'te', 'tamil nadu': 'ta',
            'kerala': 'ml', 'karnataka': 'kn', 'odisha': 'or',
            'uttar pradesh': 'hi', 'bihar': 'hi', 'madhya pradesh': 'hi',
            'rajasthan': 'hi', 'haryana': 'hi', 'himachal pradesh': 'hi',
            'delhi': 'hi', 'gujarat': 'hi', 'maharashtra': 'hi'
        }
        return {'lang': mapping.get(state, 'en'), 'state': state.title()}
    except Exception as e:
        return {'lang': 'en', 'error': str(e)}

@app.post('/api/translate-info')
async def translate_info(payload: dict):
    try:
        text = payload.get('text')
        lang = payload.get('lang', 'en')
        translated = translate_text(text, lang)
        return {'translated': translated}
    except Exception as e:
        return {'translated': payload.get('text', ''), 'error': str(e)}

@app.post('/api/get-audio-info')
async def get_audio_info(payload: dict):
    try:
        text = payload.get('text')
        lang = payload.get('lang', 'en')
        filename = create_audio(text, lang)
        if filename:
            return {'audio_url': f"/static/audio/{filename}"}
        return {'error': 'Failed to create audio'}
    except Exception as e:
        return {'error': str(e)}

# Initialize RAG vector index asynchronously on startup
@app.on_event("startup")
def startup_event():
    print("Initialising vector index...")
    plant_rag.initialize_index()

# Deep Analysis routes
@app.post("/predict_deep", response_class=HTMLResponse)
async def predict_deep(
    request: Request,
    image: UploadFile = File(None),
    filename: str = Form(None),
    language: str = Form("en"),
    lat: str = Form(None),
    lon: str = Form(None)
):
    try:
        # Parse lat/lon string to float if possible
        lat_float = None
        lon_float = None
        if lat and str(lat).strip():
            try:
                lat_float = float(lat)
            except ValueError:
                pass
        if lon and str(lon).strip():
            try:
                lon_float = float(lon)
            except ValueError:
                pass

        if filename and filename.strip():
            filename = filename.strip()
            file_path = UPLOAD_FOLDER / filename
            if not file_path.exists():
                raise HTTPException(status_code=400, detail="Uploaded file not found on server.")
        elif image and image.filename:
            # Save uploaded image
            file_ext = os.path.splitext(image.filename)[1]
            filename = f"deep_scan_{uuid.uuid4().hex}{file_ext}"
            file_path = UPLOAD_FOLDER / filename
            
            with open(file_path, "wb") as buffer:
                import shutil
                shutil.copyfileobj(image.file, buffer)
        else:
            raise HTTPException(status_code=400, detail="No image file provided.")
            
        # 1. Run CNN classification & softmax confidence
        index, class_name, confidence = predict_disease(str(file_path))
        
        # Clean label for user visibility
        clean_name = class_name.replace("___", " - ").replace("_", " ")
        
        # 2. Fetch real-time weather
        weather_data = fetch_weather(lat_float, lon_float)
        
        # 3. Calculate spread risks and recovery factors
        risk_data = calculate_risk_and_recovery(class_name, weather_data["temp"], weather_data["humidity"], weather_data["rain"])
        risk_data["confidence"] = confidence
        
        # 4. Query context from RAG vector index
        rag_context = plant_rag.retrieve_context(class_name, top_k=2)
        
        # 5. Execute Gemini Agent multi-step reasoning
        report_data = plant_agent.generate_report(clean_name, rag_context, weather_data, risk_data, language)
        
        return templates.TemplateResponse(
            request=request,
            name="report.html",
            context={
                "disease_name": clean_name,
                "confidence": confidence,
                "filename": filename,
                "weather_data": weather_data,
                "risk_data": risk_data,
                "report": report_data,
                "lang": language
            }
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return HTMLResponse(content=f"<h3>Critical internal execution error: {str(e)}</h3>", status_code=500)

@app.post("/api/chat")
async def chat_endpoint(payload: dict):
    """Grounded Chatbot assistant with dynamic memory."""
    user_msg = payload.get("message", "")
    history = payload.get("history", [])
    disease_name = payload.get("disease_name", "Unknown Disease")
    report_data = payload.get("report_data", {})
    weather_data = payload.get("weather_data", {})
    lang = payload.get("lang", "en")
    
    reply = plant_agent.chat_response(user_msg, history, disease_name, report_data, weather_data, lang)
    return JSONResponse({"reply": reply})

@app.post("/api/speech")
async def speech_endpoint(payload: dict):
    """Text to Speech helper for voice playbacks."""
    text = payload.get("text", "")
    lang = payload.get("lang", "en")
    if not text:
        return JSONResponse({"error": "Empty text inputs"}, status_code=400)
    try:
        tts = gTTS(text=text, lang=lang)
        filename = f"speech_{uuid.uuid4().hex}.mp3"
        filepath = AUDIO_FOLDER / filename
        tts.save(str(filepath))
        return JSONResponse({"audio_url": f"/static/audio/{filename}"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
