import random
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# -----------------------------
# CONFIG
# -----------------------------
# model_path = "./runs/detect/train8/weights/best.pt"  # path to your trained model
model_path = "./runs/detect/train10/weights/best.pt"
# images_dir = "./test/images"  # folder with test/validation images
images_dir = "./testing_bench"
num_images = 7  # number of random images to visualize
device = 0  # set to 'cpu' if no GPU
confidence = 0.25
# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(model_path)

# Get list of image files
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
sample_files = random.sample(image_files, num_images)

# -----------------------------
# RUN PREDICTIONS & DISPLAY
# -----------------------------
for file in sample_files:
    img_path = os.path.join(images_dir, file)

    # Predict
    results = model.predict(img_path, device=device, conf=confidence)

    # YOLO returns a list of Results objects; take first result
    result = results[0]
    print(results)
    # Get annotated image (BGR numpy array)
    annotated_img = result.plot()

    # Convert BGR to RGB for Matplotlib
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # Show
    plt.figure(figsize=(8, 8))
    plt.imshow(annotated_img_rgb)
    plt.axis("off")
    plt.title(f"Predictions: {file}")
    plt.show()
