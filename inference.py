import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import preprocess_image
import numpy as np

model = load_model('../models/mobilenetv2_weights.h5')
class_names = ['Healthy', 'Unhealthy']

def predict_image(image_path):
    img = np.expand_dims(preprocess_image(image_path), axis=0)
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    confidence = pred[0][class_idx]
    return class_names[class_idx], confidence

image_path = '../datasets/plantvillage_subset/Healthy/leaf1.jpg'
label, confidence = predict_image(image_path)
print(f"Predicted: {label} ({confidence*100:.2f}%)")
# Run inference on plant images
