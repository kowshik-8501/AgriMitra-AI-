---
title: LeafDoc AI - Plant Disease Detection
emoji: 🌿
colorFrom: green
colorTo: emerald
sdk: docker
pinned: false
app_port: 7860
---

# 🌿 LeafDoc AI: Plant Disease Detection & Crop Recommendation

LeafDoc AI is a comprehensive agricultural assistant that helps farmers and gardeners diagnose plant diseases and get crop recommendations based on soil conditions.

## 🚀 Deployment

This application is configured for **Hugging Face Spaces** using the **Docker SDK**.

### Features:
- **Disease Diagnosis**: Upload an image of a plant leaf to detect diseases.
- **Crop Recommendation**: Get AI-powered suggestions for the best crops to plant based on NPK levels and climate.
- **Multilingual Support**: Supports English, Hindi, Telugu, Tamil, and Kannada.
- **Audio Accessibility**: AI-generated voice results for better accessibility.

## 🛠️ Local Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   python app.py
   ```

## 🐳 Docker Setup

To run using Docker:
```bash
docker build -t leafdoc-ai .
docker run -p 7860:7860 leafdoc-ai
```

## 📄 License
This project is open-source and available for educational purposes.
