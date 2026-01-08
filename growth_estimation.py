import cv2
import numpy as np
import os

def estimate_leaf_area(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(c) for c in contours)
    return total_area

greenhouse_dir = '../datasets/greenhouse_images/'
for img_name in os.listdir(greenhouse_dir):
    img_path = os.path.join(greenhouse_dir, img_name)
    area = estimate_leaf_area(img_path)
    print(f"{img_name}: Leaf area = {area}")
# Leaf area estimation for plant growth
