import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import string
import sys

# 1. Setup & Alphabet
ALPHABET = string.digits + string.ascii_letters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Re-define the Model (Must match QML.py exactly)
n_qubits = 12
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class HybridQMLModel(nn.Module):
    def __init__(self, n_quantum_layers=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, n_qubits)
        )
        self.q_weights = nn.Parameter(torch.empty(n_quantum_layers, n_qubits, 3))
        self.classifier = nn.Linear(n_qubits, 62)
        
    def forward(self, x):
        features = torch.tanh(self.encoder(x)) * torch.pi 
        q_out = torch.stack(quantum_circuit(features, self.q_weights), dim=-1)
        return self.classifier(q_out.float())

# 3. Segmentation Logic (Must match your training exactly)
def segment_captcha(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    # Simple 5-way split for consistency
    chunk_w = w // 5
    segments = []
    for i in range(5):
        seg = img[:, i*chunk_w : (i+1)*chunk_w]
        seg = cv2.resize(seg, (32, 32), interpolation=cv2.INTER_AREA)
        # Normalize to 0-1 range
        seg_tensor = torch.from_numpy(seg).float().unsqueeze(0).unsqueeze(0) / 255.0
        segments.append(seg_tensor)
    return segments

def run_inference(image_path):
    # Load model
    model = HybridQMLModel().to(device)
    checkpoint = torch.load("qml_best.pth", map_location=device, weights_only=True)
    
    # Handle the compiled/uncompiled naming
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.eval()

    # Process image
    print(f"--- Predicting for: {image_path} ---")
    segments = segment_captcha(image_path)
    final_string = ""

    with torch.no_grad():
        for i, seg in enumerate(segments):
            output = model(seg.to(device))
            probabilities = torch.softmax(output, dim=1)
            conf, pred = torch.max(probabilities, 1)
            
            char = ALPHABET[pred.item()]
            final_string += char
            print(f"Char {i+1}: Predicted '{char}' | Confidence: {conf.item()*100:.1f}%")

    print(f"\nFINAL CAPTCHA PREDICTION: {final_string}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_inference(sys.argv[1])
    else:
        print("Usage: python predict.py path/to/image.png")