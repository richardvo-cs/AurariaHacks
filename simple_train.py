#!/usr/bin/env python3
"""
Simple training script for YOLOv8 parking detector
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# Set the paths
CWD = os.getcwd()
DATA_DIR = os.path.join(CWD, "data", "pklot")

# Create a new YAML file with correct paths
YAML_CONTENT = f"""
path: {DATA_DIR}
train: images/train
val: images/val

names:
  0: vacant
  1: occupied

nc: 2
"""

YAML_PATH = os.path.join(CWD, "dataset.yaml")
with open(YAML_PATH, "w") as f:
    f.write(YAML_CONTENT)

print(f"Created dataset YAML file at {YAML_PATH}")

# Prepare the model
print("Preparing YOLOv8 model...")
os.makedirs("models", exist_ok=True)
model_path = os.path.join(CWD, "models", "yolov8s.pt")

if not os.path.exists(model_path):
    print("Downloading YOLOv8s model...")
    model = YOLO("yolov8s.pt")
    model.save(model_path)
else:
    print("Loading existing YOLOv8s model...")
    model = YOLO(model_path)

# Train the model
print("Starting training...")
print(f"Using dataset at {DATA_DIR}")
print(f"Training with images from {os.path.join(DATA_DIR, 'images/train')}")
print(f"Validating with images from {os.path.join(DATA_DIR, 'images/val')}")

results = model.train(
    data=YAML_PATH,
    epochs=50,
    imgsz=640,
    batch=16,
    name="parking_detector",
    project="runs",
    exist_ok=True,
    patience=10,
)

# Validate the model
print("Validating model...")
model = YOLO(Path("runs/parking_detector/weights/best.pt"))
results = model.val(data=YAML_PATH)

print(f"Training complete! Best model saved at runs/parking_detector/weights/best.pt")
print(f"Validation results: {results}") 