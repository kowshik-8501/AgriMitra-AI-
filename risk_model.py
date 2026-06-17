import httpx

def fetch_weather(lat: float, lon: float) -> dict:
    """
    Fetches real-time weather data from Open-Meteo API based on coordinates.
    If coordinates are missing or API fails, returns default placeholder values.
    """
    default_weather = {
        "temp": 27.5,
        "humidity": 65.0,
        "rain": 0.0,
        "soil_moisture": "Moderate",
        "location_status": "Default (Forecast Mode)"
    }
    
    if lat is None or lon is None:
        return default_weather
        
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,rain&timezone=auto"
        headers = {"User-Agent": "PlantAI2.0/1.0"}
        
        # Keep timeout small (3s) to prevent blocking page loads
        with httpx.Client(headers=headers, timeout=3.0) as client:
            response = client.get(url)
            if response.status_code == 200:
                data = response.json()
                current = data.get("current", {})
                
                # Derive soil moisture proxy from recent rain
                rain_val = float(current.get("rain", 0.0))
                soil_m = "Wet" if rain_val > 5.0 else ("Dry" if rain_val == 0.0 and float(current.get("temperature_2m", 25)) > 30 else "Moderate")
                
                return {
                    "temp": float(current.get("temperature_2m", 27.5)),
                    "humidity": float(current.get("relative_humidity_2m", 65.0)),
                    "rain": rain_val,
                    "soil_moisture": soil_m,
                    "location_status": f"Live Data ({lat:.3f}, {lon:.3f})"
                }
    except Exception as e:
        print(f"Failed to fetch live weather: {e}")
        
    return default_weather

def calculate_risk_and_recovery(disease_name: str, temp: float, humidity: float, rain: float) -> dict:
    """
    Determines spread risk category and recovery probability using agricultural rules.
    """
    disease_lower = disease_name.lower()
    
    # Healthy plants have no spread risk and 100% recovery
    if "healthy" in disease_lower or "background" in disease_lower:
        return {
            "spread_risk": "Low",
            "spread_risk_score": 10,
            "recovery_prob": 100,
            "disease_type": "None (Healthy Crop)",
            "analysis": "The plant is healthy. Maintain standard cultural practices and optimal fertilization."
        }
        
    # Categorize disease type
    is_fungal = any(x in disease_lower for x in ["rust", "blight", "scab", "rot", "mildew", "mold", "scorch", "spot"])
    is_viral = any(x in disease_lower for x in ["virus", "curl", "mosaic"])
    is_pest = "mite" in disease_lower
    
    spread_score = 50  # base score
    disease_type = "Bacterial Disease"
    
    if is_fungal:
        disease_type = "Fungal Infection"
        # Fungi thrive in warm, highly humid/rainy environments
        if humidity > 80:
            spread_score += 25
        if 18 <= temp <= 28:
            spread_score += 15
        if rain > 2.0:
            spread_score += 10
            
    elif is_viral:
        disease_type = "Viral Pathogen"
        # Viruses spread via insects (whiteflies, aphids) which multiply in hot, dry conditions
        if temp > 28:
            spread_score += 20
        if humidity < 60:
            spread_score += 15
            
    elif is_pest:
        disease_type = "Pest Infestation"
        # Spider mites multiply extremely fast in hot and dry climates
        if temp > 30:
            spread_score += 25
        if humidity < 50:
            spread_score += 15
            
    else:
        # Default Bacterial Spot/Greening rules
        if temp > 26 and humidity > 75:
            spread_score += 30

    # Ensure bounds
    spread_score = max(10, min(98, spread_score))
    
    if spread_score >= 75:
        risk_cat = "High"
    elif spread_score >= 45:
        risk_cat = "Moderate"
    else:
        risk_cat = "Low"
        
    # Calculate recovery probability
    # Fast-spreading diseases lower the recovery probability
    rec_prob = 100 - (spread_score * 0.7)
    
    # viral diseases have lower cure rates than fungal/pests
    if is_viral:
        rec_prob -= 15
        
    # Ensure realistic range
    rec_prob = max(15, min(95, round(rec_prob, 2)))
    
    # Detailed risk analysis text
    analysis_text = (
        f"This {disease_type} is currently in a "
        f"{'highly favorable' if risk_cat == 'High' else ('moderately favorable' if risk_cat == 'Moderate' else 'mostly unfavorable')} "
        f"climate zone. Temp: {temp}°C, Humidity: {humidity}%, and Rain: {rain}mm. "
    )
    if is_fungal:
        analysis_text += "High humidity keeps leaves wet, which accelerates fungal spore germination."
    elif is_viral:
        analysis_text += "Warm temperatures promote fast vector breeding, speeding up viral transmission."
    elif is_pest:
        analysis_text += "Dry and warm microclimates are ideal for mite population growth."
    else:
        analysis_text += "Wet leaf surfaces and warm winds encourage bacterial multiplication."
        
    return {
        "spread_risk": risk_cat,
        "spread_risk_score": spread_score,
        "recovery_prob": rec_prob,
        "disease_type": disease_type,
        "analysis": analysis_text
    }
