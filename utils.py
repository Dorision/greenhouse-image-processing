import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = img_to_array(image) / 255.0
    return image

def load_dataset(dataset_dir, target_size=(224, 224)):
    images = []
    labels = []
    classes = sorted(os.listdir(dataset_dir))
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_dir, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = preprocess_image(img_path, target_size)
            images.append(img)
            labels.append(idx)
    return np.array(images), np.array(labels), classes
# Preprocessing and dataset loading utilities
