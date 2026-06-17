import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

import CNN

# Load model path relatives
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, 'plant_disease_model_1_latest.pt'))

# Initialize and cache model
model = CNN.CNN(39)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Retrieve class mappings
idx_to_classes = CNN.idx_to_classes

def predict_disease(image_path: str):
    """
    Predicts plant disease from an image path.
    Returns:
        index: (int) the class index
        class_name: (str) the raw folder class name
        confidence: (float) confidence percentage (0-100)
    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    
    with torch.no_grad():
        output = model(input_data)
        probabilities = F.softmax(output, dim=1).numpy()[0]
        
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index] * 100)
    class_name = idx_to_classes.get(index, "Unknown")
    
    return index, class_name, round(confidence, 2)
