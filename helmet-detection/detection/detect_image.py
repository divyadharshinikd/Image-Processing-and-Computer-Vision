import cv2
import os
import glob
from ultralytics import YOLO

def detect_on_image():
    # Model path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'runs', 'detect', 'train-2', 'weights', 'best.pt')
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    # Path to validation images
    images_dir = os.path.join(current_dir, '..', 'dataset', 'images', 'val')
    images = glob.glob(os.path.join(images_dir, '*.png'))
    
    if not images:
        print("No images found in dataset/images/val")
        return
        
    # Take the first image
    image_path = images[0]
    print(f"Running detection on image: {image_path}")
    
    # Run prediction
    # We save the output to the detection folder so you can view it
    results = model(image_path, save=True, project=os.path.join(current_dir, 'output'), name='test_run', exist_ok=True)
    
    output_path = os.path.join(current_dir, 'output', 'test_run', os.path.basename(image_path))
    print(f"Detection complete! Result saved to: {output_path}")

if __name__ == '__main__':
    detect_on_image()
