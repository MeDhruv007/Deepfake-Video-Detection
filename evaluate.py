import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_data
from model import Meso4
import os

def evaluate_model():
    print("Loading test data...")
    _, _, test_ds = load_data()
    
    print("Building model and loading weights...")
    model = Meso4(input_shape=(256, 256, 3))
    
    if not os.path.exists('weights/meso4_best.weights.h5'):
        print("Error: Weights file not found! Please run 'python train.py' first to train the model.")
        return

    model.load_weights('weights/meso4_best.weights.h5')
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    
    print("\nEvaluating on test set...")
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    print("\nGenerating predictions for Confusion Matrix...")
    y_pred = []
    y_true = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(preds)
        y_true.extend(labels.numpy())
        
    y_pred = np.array(y_pred) > 0.5
    y_true = np.array(y_true)
    
    print("\n------------------------------")
    print("Classification Report:")
    print("------------------------------")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - Meso4 DeepFake Detector')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("\nCONFUSION MATRIX SAVED => 'confusion_matrix.png'")

if __name__ == "__main__":
    evaluate_model()
