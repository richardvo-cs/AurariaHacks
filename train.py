#!/usr/bin/env python3
"""
Train a YOLOv8 model for parking space occupancy detection.
This script loads the YOLOv8s model and fine-tunes it on the PKLot dataset.
"""

import os
import sys
from ultralytics import YOLO
from pathlib import Path

def main():
    # Get current directory
    current_dir = os.getcwd()
    
    # Check if GPU is available
    print("Checking for GPU...")
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if gpu_available else 0
        device_name = torch.cuda.get_device_name(0) if gpu_available and device_count > 0 else "None"
        print(f"GPU available: {gpu_available}, Device count: {device_count}, Device name: {device_name}")
    except ImportError:
        print("PyTorch not found, running on CPU")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Create necessary directories for dataset compatibility
    data_dir = os.path.join(current_dir, "datasets", "data", "pklot")
    os.makedirs(os.path.join(data_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "labels", "val"), exist_ok=True)
    
    # Create symbolic links to the actual data
    source_images_train = os.path.join(current_dir, "data", "pklot", "images", "train")
    source_images_val = os.path.join(current_dir, "data", "pklot", "images", "val")
    source_labels_train = os.path.join(current_dir, "data", "pklot", "labels", "train")
    source_labels_val = os.path.join(current_dir, "data", "pklot", "labels", "val")
    
    target_images_train = os.path.join(data_dir, "images", "train")
    target_images_val = os.path.join(data_dir, "images", "val")
    target_labels_train = os.path.join(data_dir, "labels", "train")
    target_labels_val = os.path.join(data_dir, "labels", "val")
    
    print(f"Creating symbolic links for dataset compatibility...")
    
    # Copy or link files for training images
    for image_file in os.listdir(source_images_train):
        if image_file.endswith('.jpg') or image_file.endswith('.jpeg') or image_file.endswith('.png'):
            source_path = os.path.join(source_images_train, image_file)
            target_path = os.path.join(target_images_train, image_file)
            if not os.path.exists(target_path):
                try:
                    # Try symbolic link first
                    os.symlink(source_path, target_path)
                except:
                    # Fall back to copy if symlink fails
                    import shutil
                    shutil.copy2(source_path, target_path)
    
    # Copy or link files for validation images
    for image_file in os.listdir(source_images_val):
        if image_file.endswith('.jpg') or image_file.endswith('.jpeg') or image_file.endswith('.png'):
            source_path = os.path.join(source_images_val, image_file)
            target_path = os.path.join(target_images_val, image_file)
            if not os.path.exists(target_path):
                try:
                    os.symlink(source_path, target_path)
                except:
                    import shutil
                    shutil.copy2(source_path, target_path)
    
    # Copy or link files for training labels
    for label_file in os.listdir(source_labels_train):
        if label_file.endswith('.txt'):
            source_path = os.path.join(source_labels_train, label_file)
            target_path = os.path.join(target_labels_train, label_file)
            if not os.path.exists(target_path):
                try:
                    os.symlink(source_path, target_path)
                except:
                    import shutil
                    shutil.copy2(source_path, target_path)
    
    # Copy or link files for validation labels
    for label_file in os.listdir(source_labels_val):
        if label_file.endswith('.txt'):
            source_path = os.path.join(source_labels_val, label_file)
            target_path = os.path.join(target_labels_val, label_file)
            if not os.path.exists(target_path):
                try:
                    os.symlink(source_path, target_path)
                except:
                    import shutil
                    shutil.copy2(source_path, target_path)
    
    print("Dataset preparation complete")
    
    # Create a dataset YAML file in the expected location
    yaml_content = f"""
path: {data_dir}
train: images/train
val: images/val

names:
  0: vacant
  1: occupied

nc: 2
"""
    
    dataset_yaml_path = os.path.join(current_dir, "datasets_yaml.yaml")
    with open(dataset_yaml_path, "w") as f:
        f.write(yaml_content)
    
    # Download YOLOv8s model if not present
    model_path = Path("models/yolov8s.pt")
    if not model_path.exists():
        print("Downloading YOLOv8s model...")
        model = YOLO("yolov8s.pt")
        model.save(model_path)
    else:
        print("Loading existing YOLOv8s model...")
        model = YOLO(model_path)
    
    # Train the model
    print("Starting training...")
    results = model.train(
        data=dataset_yaml_path,
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
    results = model.val(data=dataset_yaml_path)
    
    print(f"Training complete! Best model saved at runs/parking_detector/weights/best.pt")
    print(f"Validation results: {results}")

if __name__ == "__main__":
    main() 