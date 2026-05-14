import os
import uuid
import requests
# pyrefly: ignore [missing-import]
from flask import Flask, render_template, request, url_for, send_file, jsonify
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
import torchvision.transforms.functional as TF
import CNN
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
import pandas as pd
# pyrefly: ignore [missing-import]
from deep_translator import GoogleTranslator
import time
from gtts import gTTS
import joblib

def translate_text(text, target_lang):
    try:
        if not text or target_lang == 'en': return text
        lang_map = {
            'hi': 'hi', 'te': 'te', 'ta': 'ta', 'kn': 'kn', 
            'ml': 'ml', 'or': 'or'
        }
        target = lang_map.get(target_lang, 'hi')
        translated = GoogleTranslator(source='auto', target=target).translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def sanitize_text(text):
    """Clean text for gTTS: remove HTML tags and extra whitespace."""
    if not text:
        return ""
    import re
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text

def cleanup_old_audio(folder, max_age_seconds=3600):
    """Remove audio files older than max_age_seconds."""
    try:
        now = time.time()
        for f in os.listdir(folder):
            filepath = os.path.join(folder, f)
            if os.path.isfile(filepath) and f.endswith('.mp3'):
                if now - os.path.getmtime(filepath) > max_age_seconds:
                    os.remove(filepath)
    except Exception as e:
        print(f"Cleanup error: {e}")

def create_audio(text, lang_code):
    try:
        # Cleanup old files first
        cleanup_old_audio(AUDIO_FOLDER)
        
        # Sanitize text
        clean_text = sanitize_text(text)
        if not clean_text:
            return None
            
        lang_map = {
            'hi': 'hi', 'te': 'te', 'ta': 'ta', 'kn': 'kn', 
            'en': 'en', 'ml': 'ml', 'or': 'or'
        }
        target = lang_map.get(lang_code, 'en')
        
        print(f"Generating audio for lang: {target}, text length: {len(clean_text)}")
        
        tts = gTTS(text=clean_text, lang=target)
        filename = f"speech_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(AUDIO_FOLDER, filename)
        tts.save(filepath)
        
        if os.path.exists(filepath):
            return filename
        return None
    except Exception as e:
        print(f"Audio creation error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Ensure the uploads and audio directories exist
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
AUDIO_FOLDER = os.path.join(BASE_DIR, 'static', 'audio')

for folder in [UPLOAD_FOLDER, AUDIO_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

disease_info = pd.read_csv(os.path.join(BASE_DIR, "disease_info.csv"), encoding='cp1252')
supplement_info = pd.read_csv(os.path.join(BASE_DIR, "supplement_info.csv"), encoding='cp1252')

model = CNN.CNN(39)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "plant_disease_model_1_latest.pt"), map_location=torch.device('cpu')))
model.eval()

def prediction(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

class YieldPredictor:
    def __init__(self):
        self.model_path = os.path.join(BASE_DIR, 'models', 'yield_model.pkl')
        self.le_dist_path = os.path.join(BASE_DIR, 'models', 'le_district.pkl')
        self.le_crop_path = os.path.join(BASE_DIR, 'models', 'le_crop.pkl')
        self.data_dir = os.path.join(BASE_DIR, 'data')
        self.load_artifacts()

    def load_artifacts(self):
        try:
            self.model = joblib.load(self.model_path)
            self.le_district = joblib.load(self.le_dist_path)
            self.le_crop = joblib.load(self.le_crop_path)
            self.history = pd.read_csv(os.path.join(self.data_dir, 'historical_yield.csv'))
        except Exception as e:
            print(f"Error loading AI artifacts: {e}")
            self.model = None

    def predict(self, input_data):
        try:
            crop = input_data.get('crop', 'Rice')
            district = input_data.get('district', 'Guntur')
            
            try:
                dist_enc = self.le_district.transform([district])[0]
            except:
                dist_enc = 0
                
            try:
                crop_enc = self.le_crop.transform([crop])[0]
            except:
                crop_enc = 0

            rain_val = float(input_data.get('rainfall', 1000))
            temp_val = float(input_data.get('temp', 28))
            
            features = np.array([[
                dist_enc, 
                crop_enc, 
                rain_val, 
                temp_val,
                float(input_data.get('n', 80)),
                float(input_data.get('p', 40)),
                float(input_data.get('k', 40))
            ]])

            if self.model:
                prediction = self.model.predict(features)[0]
            else:
                prediction = 3.5
                
            hist_matches = self.history[self.history['District'].str.contains(district, case=False)]
            base_val = float(hist_matches['Yield'].mean()) if not hist_matches.empty else float(prediction)
            
            trend = [float(base_val * (1 + np.random.uniform(-0.1, 0.1))) for _ in range(5)]
            trend.append(float(round(prediction, 2)))
            
            return float(round(prediction, 2)), trend

        except Exception as e:
            print(f"Prediction logic error: {e}")
            return 3.5, [3.0, 3.2, 3.1, 3.4, 3.3, 3.5]

yield_predictor = YieldPredictor()



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
        if not recs: recs.append("Your soil NPK levels are optimal for this crop. Maintain organic matter.")
        return recs

fertilizer_advisor = FertilizerAdvisor()

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        try:
            image = request.files['image']
            filename = image.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            image.save(file_path)
            
            pred = prediction(file_path)
            pred = int(pred)
            if pred >= len(disease_info):
                pred = 0
                
            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]
            
            lang = request.form.get('language', 'en')
            lang_note = translate_text(title, lang) if lang != 'en' else None
            
            # Full Translation for display
            display_desc = translate_text(description, lang) if lang != 'en' else description
            display_prevent = translate_text(prevent, lang) if lang != 'en' else prevent
            
            # Audio Generation (AI Telugu Voice / Native Voice)
            audio_text = f"Diagnosis result: {lang_note if lang_note else title}. Details: {display_desc}. Prevention: {display_prevent}."
            audio_file = create_audio(audio_text, lang)

            return render_template('submit.html', title=title, desc=display_desc, prevent=display_prevent,
                                   image_url=image_url, pred=pred, sname=supplement_name, simage=supplement_image_url, buy_link=supplement_buy_link,
                                   selected_lang=lang, lang_note=lang_note, audio_file=audio_file)
        except Exception as e:
            import traceback
            print("Error during submission:")
            traceback.print_exc()
            return f"An error occurred: {str(e)}. Please check the server logs for details.", 500

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), disease=list(disease_info['disease_name']), buy=list(supplement_info['buy link']))

@app.route('/contact')
def contact_page():
    return render_template('contact-us.html')

@app.route('/documentation')
def documentation():
    return send_file(os.path.join(BASE_DIR, 'LICENSE'), mimetype='text/plain')

import pickle

# Load the trained crop recommendation model
crop_model_path = os.path.join(BASE_DIR, 'crop_model.pkl')
with open(crop_model_path, 'rb') as f:
    crop_model = pickle.load(f)

CROP_DETAILS = {
    'rice': "Rice is a staple cereal grain and the primary food source for a large part of the world's population. It is a semi-aquatic plant that requires significant water and a warm climate.",
    'maize': "Maize, also known as corn, is a versatile cereal grain. It is used for human food, livestock feed, and industrial products like ethanol.",
    'chickpea': "Chickpea is a highly nutritious legume, a key source of protein in many diets. It is drought-tolerant and improves soil fertility by fixing nitrogen.",
    'kidneybeans': "Kidney beans are a variety of common beans, named for their resemblance to human kidneys. They are rich in protein, fiber, and essential minerals.",
    'pigeonpeas': "Pigeonpeas are a perennial legume often grown for their edible seeds. They are highly drought-resistant and popular in tropical and subtropical regions.",
    'mothbeans': "Mothbeans are small, drought-resistant legumes primarily grown in arid regions. They are a good source of protein and can grow in poor soil conditions.",
    'mungbean': "Mungbean is a small green legume used in both sweet and savory dishes. It is fast-growing and requires relatively less water.",
    'blackgram': "Blackgram, or Urad Dal, is a popular pulse in South Asia. It is valued for its high protein content and is often used in fermented foods like idli and dosa.",
    'lentil': "Lentils are small, lens-shaped legumes. They are easy to cook and are an excellent source of plant-based protein and iron.",
    'pomegranate': "Pomegranate is a fruit-bearing shrub known for its antioxidant-rich red seeds. It thrives in hot, dry climates and well-drained soil.",
    'banana': "Banana is one of the most widely consumed fruits globally. The plant requires a tropical climate with high humidity and consistent moisture.",
    'mango': "Mango, the 'king of fruits,' is a tropical stone fruit. Mango trees prefer distinct wet and dry seasons and deep, fertile soil.",
    'grapes': "Grapes are used for fresh consumption, juice, and winemaking. They require well-drained soil and specific pruning and support systems.",
    'watermelon': "Watermelon is a refreshing summer fruit with high water content. It grows on long vines and requires plenty of sunlight and warm temperatures.",
    'muskmelon': "Muskmelon is a sweet, fragrant fruit similar to cantaloupe. It thrives in warm weather and sandy, well-drained soil.",
    'apple': "Apples are one of the most popular temperate fruits. They require a cool climate with a certain number of 'chilling hours' to fruit properly.",
    'orange': "Oranges are citrus fruits rich in Vitamin C. They grow best in subtropical to tropical climates with moderate rainfall.",
    'papaya': "Papaya is a fast-growing tropical fruit tree. It produces fruit year-round and requires well-drained soil and protection from strong winds.",
    'coconut': "Coconut palm is a versatile tree found in coastal tropical regions. Every part of the tree, from the fruit to the leaves, has economic value.",
    'cotton': "Cotton is a major fiber crop used in the textile industry. It requires a long frost-free period, plenty of sunshine, and moderate rainfall.",
    'jute': "Jute is a long, soft, shiny vegetable fiber that can be spun into coarse, strong threads. It thrives in hot and humid climates with high rainfall.",
    'coffee': "Coffee is a popular beverage made from roasted seeds. Coffee plants prefer high altitudes, consistent temperatures, and moderate rainfall."
}

CROP_CENTERS = {
    'rice': [80, 40, 40, 23, 82, 6.5, 202],
    'maize': [100, 48, 20, 22, 65, 6.2, 84],
    'chickpea': [40, 67, 79, 18, 16, 7.3, 80],
    'kidneybeans': [20, 76, 20, 20, 21, 5.7, 105],
    'pigeonpeas': [20, 67, 20, 27, 48, 5.7, 149],
    'mothbeans': [21, 48, 20, 28, 53, 6.8, 51],
    'mungbean': [20, 47, 20, 28, 88, 6.7, 48],
    'blackgram': [40, 67, 20, 29, 65, 7.1, 67],
    'lentil': [20, 67, 19, 24, 64, 6.9, 45],
    'pomegranate': [20, 18, 40, 21, 90, 6.4, 107],
    'banana': [100, 80, 50, 27, 80, 5.9, 104],
    'mango': [20, 27, 30, 29, 50, 5.7, 94],
    'grapes': [23, 125, 200, 23, 81, 6.0, 69],
    'watermelon': [100, 18, 50, 25, 85, 6.4, 50],
    'muskmelon': [100, 18, 50, 28, 92, 6.3, 24],
    'apple': [20, 137, 199, 22, 92, 5.9, 112],
    'orange': [20, 10, 10, 22, 92, 7.0, 110],
    'papaya': [50, 50, 50, 33, 92, 6.7, 142],
    'coconut': [20, 10, 30, 27, 94, 5.9, 175],
    'cotton': [117, 46, 19, 23, 80, 6.9, 40],
    'jute': [80, 46, 40, 24, 79, 6.7, 174],
    'coffee': [101, 28, 30, 25, 57, 6.7, 158]
}

def get_crop_reason(crop, n, p, k, temp, hum, ph, rain):
    crop_lower = crop.lower()
    if crop_lower not in CROP_CENTERS:
        return f"This crop is well-suited for the combination of nutrients and climate in your area."
    
    center = CROP_CENTERS[crop_lower]
    reasons = []
    
    # Simple logic to find the most "defining" characteristics for this prediction
    if abs(n - center[0]) < 20: reasons.append(f"the Nitrogen level ({n}) is ideal")
    if abs(p - center[1]) < 20: reasons.append(f"Phosphorus ({p}) is in the optimal range")
    if abs(k - center[2]) < 20: reasons.append(f"Potassium ({k}) supports its growth")
    if abs(temp - center[3]) < 5: reasons.append(f"the temperature ({temp}°C) is perfect")
    if abs(hum - center[4]) < 10: reasons.append(f"humidity ({hum}%) matches its needs")
    if abs(ph - center[5]) < 0.5: reasons.append(f"soil pH ({ph}) is just right")
    if abs(rain - center[6]) < 30: reasons.append(f"rainfall ({rain}mm) provides the necessary moisture")
    
    if not reasons:
        return f"This crop is the best match for your overall soil profile and weather conditions."
    
    return f"{crop.capitalize()} is recommended because " + ", ".join(reasons[:3]) + "."

@app.route('/crop', methods=['GET', 'POST'])
def crop_page():
    crop = None
    crop_desc = None
    crop_reason = None
    language_note = None
    audio_file = None
    
    if request.method == 'POST':
        try:
            n = float(request.form.get('nitrogen', 0))
            p = float(request.form.get('phosphorus', 0))
            k = float(request.form.get('potassium', 0))
            temp = float(request.form.get('temperature', 0))
            hum = float(request.form.get('humidity', 0))
            ph = float(request.form.get('ph', 0))
            rain = float(request.form.get('rainfall', 0))
            lang = request.form.get('language', 'en')
            
            # Predict using the loaded model
            input_features = np.array([[n, p, k, temp, hum, ph, rain]])
            prediction = crop_model.predict(input_features)[0]
            crop_name = str(prediction).capitalize()
            crop = crop_name
            
            # Get Description and Reason
            raw_desc = CROP_DETAILS.get(prediction.lower(), "Information not available.")
            raw_reason = get_crop_reason(prediction, n, p, k, temp, hum, ph, rain)
            
            # Translate if necessary
            if lang != 'en':
                language_note = translate_text(crop_name, lang)
                crop_desc = translate_text(raw_desc, lang)
                crop_reason = translate_text(raw_reason, lang)
            else:
                crop_desc = raw_desc
                crop_reason = raw_reason
            
            # Audio Generation
            recommendation_text = f"The AI recommends planting {language_note if language_note else crop_name}. {raw_desc} {raw_reason}"
            if lang != 'en':
                # Translate the full audio text for natural flow
                audio_text = translate_text(recommendation_text, lang)
            else:
                audio_text = recommendation_text
                
            audio_file = create_audio(audio_text, lang)
                
        except Exception as e:
            print(f"Error in crop prediction: {e}")
            crop = "Error in prediction"

    return render_template('crop.html', crop=crop, crop_desc=crop_desc, crop_reason=crop_reason, 
                           language_note=language_note, audio_file=audio_file)



@app.route('/yield', methods=['GET', 'POST'])
def yield_prediction():
    result = None
    input_data = {}
    trend_data = []
    lang = 'en'
    audio_file = None
    trans_result = None
    
    if request.method == 'POST':
        try:
            lang = request.form.get('language', 'en')
            input_data = {
                'district': request.form.get('district', '').strip(),
                'crop': request.form.get('crop'),
                'n': float(request.form.get('n', 0)),
                'p': float(request.form.get('p', 0)),
                'k': float(request.form.get('k', 0)),
                'temp': float(request.form.get('temp', 0)),
                'rainfall': float(request.form.get('rainfall', 0))
            }
            
            prediction, trend = yield_predictor.predict(input_data)
            result = prediction
            trend_data = trend
            
            # Regional Comparison Data
            hist = yield_predictor.history
            crop_matches = hist[hist['Crop'].str.contains(input_data['crop'], case=False)]
            
            if crop_matches.empty:
                # Fallback to top districts overall
                top_regions = hist.groupby('District')['Yield'].mean().sort_values(ascending=False).head(5)
            else:
                top_regions = crop_matches.groupby('District')['Yield'].mean().sort_values(ascending=False).head(5)
                
            regional_data = {
                'labels': top_regions.index.tolist(),
                'chart_values': [round(float(v), 2) for v in top_regions.values],
                'current_district': input_data['district'],
                'current_val': float(prediction)
            }

            # Analysis Text
            if prediction > 4.5:
                status = "Excellent"
                advice = "Conditions are perfect. Consider early market booking."
            elif prediction > 3.0:
                status = "Good"
                advice = "Standard maintenance will secure this harvest."
            else:
                status = "Average"
                advice = "Review soil nutrients and irrigation frequency."

            report_text = f"The predicted yield for {input_data['crop']} in {input_data['district']} is {prediction} tons per hectare. Status: {status}. {advice}"
            trans_result = translate_text(report_text, lang)
            audio_file = create_audio(trans_result, lang)
            
        except Exception as e:
            print(f"Prediction Error: {e}")
            result = "Error in calculation"

    return render_template('yield.html', 
                         result=result, 
                         input_data=input_data, 
                         trend=trend_data,
                         regional_data=regional_data if 'regional_data' in locals() else None,
                         lang=lang,
                         trans_result=trans_result,
                         audio_file=audio_file)


@app.route('/fertilizer', methods=['GET', 'POST'])
def fertilizer_recommendation():
    recs = None
    input_data = {}
    lang = 'en'
    audio_file = None
    trans_recs = []
    
    if request.method == 'POST':
        try:
            lang = request.form.get('language', 'en')
            n = float(request.form.get('n', 0))
            p = float(request.form.get('p', 0))
            k = float(request.form.get('k', 0))
            crop = request.form.get('crop', 'Paddy')
            input_data = {'n': n, 'p': p, 'k': k, 'crop': crop}
            recs = fertilizer_advisor.recommend(n, p, k, crop)
            
            # Translation and Audio
            if lang != 'en':
                trans_recs = [translate_text(r, lang) for r in recs]
                audio_text = ". ".join(trans_recs)
            else:
                audio_text = ". ".join(recs)
            
            audio_file = create_audio(audio_text, lang)
            
        except Exception as e:
            print(f"Error in fertilizer rec: {e}")
            recs = ["Error processing request"]
            
    crops = ['Paddy', 'Maize', 'Sugarcane', 'Cotton', 'Wheat', 'Rice']
    return render_template('fertilizer.html', recs=recs, data=input_data, crops=crops, 
                           selected_lang=lang, trans_recs=trans_recs, audio_file=audio_file)

@app.route('/explore')
def explore():
    diseases = disease_info.to_dict(orient='records')
    return render_template('explore.html', diseases=diseases)

@app.route('/api/detect-location', methods=['POST'])
def detect_location():
    try:
        data = request.get_json()
        lat = data.get('lat')
        lon = data.get('lon')
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10"
        headers = {'User-Agent': 'CropCareAI/1.0'}
        response = requests.get(url, headers=headers).json()
        addr = response.get('address', {})
        state = addr.get('state', '').lower()
        
        mapping = {
            'andhra pradesh': 'te', 'telangana': 'te', 'tamil nadu': 'ta',
            'kerala': 'ml', 'karnataka': 'kn', 'odisha': 'or',
            'uttar pradesh': 'hi', 'bihar': 'hi', 'madhya pradesh': 'hi',
            'rajasthan': 'hi', 'haryana': 'hi', 'himachal pradesh': 'hi',
            'delhi': 'hi', 'gujarat': 'hi', 'maharashtra': 'hi'
        }
        
        return jsonify({'lang': mapping.get(state, 'en'), 'state': state.title()})
    except Exception as e:
        return jsonify({'lang': 'en', 'error': str(e)})

@app.route('/api/translate-info', methods=['POST'])
def translate_info():
    try:
        data = request.get_json()
        text = data.get('text')
        lang = data.get('lang', 'en')
        translated = translate_text(text, lang)
        return jsonify({'translated': translated})
    except Exception as e:
        return jsonify({'translated': text, 'error': str(e)})

@app.route('/api/get-audio-info', methods=['POST'])
def get_audio_info():
    try:
        data = request.get_json()
        text = data.get('text')
        lang = data.get('lang', 'en')
        filename = create_audio(text, lang)
        if filename:
            return jsonify({'audio_url': url_for('static', filename='audio/' + filename)})
        return jsonify({'error': 'Failed to create audio'})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)