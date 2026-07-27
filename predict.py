import argparse
import numpy as np
import tensorflow as tf
from model import Meso4
import cv2
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence TF logs

def extract_face(image):
    """
    Detects the largest face in the image and crops it with a slight margin.
    If no face is found, returns None.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None 
        
    # Get the the largest face by area
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    
    # DeepFake models usually need a slight margin around the box to see edges
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.2)
    
    start_x = max(0, x - margin_x)
    start_y = max(0, y - margin_y)
    end_x = min(image.shape[1], x + w + margin_x)
    end_y = min(image.shape[0], y + h + margin_y)
    
    cropped_face = image[start_y:end_y, start_x:end_x]
    return cropped_face

def load_video_faces(video_path, max_faces=20):
    """
    Reads frames from the video and returns a list of cropped faces.
    Scans up to 100 frames to find up to max_faces.
    """
    cap = cv2.VideoCapture(video_path)
    faces = []
    
    # Read up to 100 frames looking for faces
    for _ in range(100):
        ret, frame = cap.read()
        if not ret or len(faces) >= max_faces:
            break
        
        # Keep consistent with training (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = extract_face(frame_rgb)
        if face is not None:
            faces.append(face)
            
    cap.release()
    return faces

def predict(media_path):
    if not os.path.exists('weights/meso4_best.weights.h5'):
        print("Error: Model weights not found! Please run 'python train.py' to train the model first.")
        sys.exit(1)

    # Initialize model
    model = Meso4(input_shape=(256, 256, 3))
    model.load_weights('weights/meso4_best.weights.h5')
    
    # Load and crop
    if media_path.lower().endswith(('.mp4', '.avi', '.mov')):
        print(f"Scanning video frames to detect faces: {media_path}...")
        faces = load_video_faces(media_path)
    else:
        print(f"Loading image and extracting face: {media_path}...")
        image = cv2.imread(media_path)
        if image is None:
            print("Error: Could not read image.")
            sys.exit(1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face = extract_face(image)
        faces = [face] if face is not None else []
        
    if not faces:
        print("\nERROR: Could not detect any faces in the provided media!")
        print("MesoNet requires localized faces to analyze mesoscopic artifacts. Please try a different file.")
        sys.exit(1)
        
    print(f"Found {len(faces)} faces. Running Meso-4 inference...")
    
    preds = []
    for face in faces:
        # Prepare for model
        face = cv2.resize(face, (256, 256))
        face = face / 255.0
        face = np.expand_dims(face, axis=0) # Add Batch dimension
        
        p = model.predict(face, verbose=0)[0][0]
        preds.append(p)
    
    # Average prediction
    pred = sum(preds) / len(preds)
    
    # Fake = 1, Real = 0
    confidence = pred if pred > 0.5 else 1 - pred
    label = "FAKE" if pred > 0.5 else "REAL"
    
    print("\n==============================", flush=True)
    print(f" File: {os.path.basename(media_path)}", flush=True)
    print(f" Prediction: {label}", flush=True)
    print(f" Confidence: {confidence:.2%}", flush=True)
    print("==============================\n", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict DeepFake from image or video using Meso-4")
    parser.add_argument('path', type=str, help="Path to video or image")
    args = parser.parse_args()
    predict(args.path)
