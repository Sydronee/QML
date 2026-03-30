import os
import glob
import cv2
import torch
import torch.nn as nn
import pennylane as qml
import string
from tqdm import tqdm
import numpy as np
import random

# 1. Setup & Alphabet
ALPHABET = string.digits + string.ascii_letters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Model Definition (Must match your QML.py exactly)
n_qubits = 12
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class HybridQMLModel(nn.Module):
    def __init__(self, n_quantum_layers=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2), 
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), 
            nn.Flatten(), nn.Linear(64 * 4 * 4, 512), nn.ReLU(), nn.Linear(512, n_qubits)
        )
        self.q_weights = nn.Parameter(torch.empty(n_quantum_layers, n_qubits, 3))
        self.classifier = nn.Linear(n_qubits, 62)
        
    def forward(self, x):
        features = torch.tanh(self.encoder(x)) * np.pi 
        q_out = torch.stack(quantum_circuit(features, self.q_weights), dim=-1)
        return self.classifier(q_out.float())

def run_full_evaluation(data_dir="Large_Captcha_Dataset", limit=5000):
    # Load Model
    model = HybridQMLModel().to(device)
    if os.path.exists("qml_best.pth"):
        checkpoint = torch.load("qml_best.pth", map_location=device, weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state_dict.items()})
    model.eval()

    image_paths = glob.glob(os.path.join(data_dir, "*.png"))
    random.seed(42) # Keeps it consistent so you can compare later
    random.shuffle(image_paths)
    if limit:
        image_paths = image_paths[:limit]
    
    print(f"Starting evaluation on {len(image_paths)} images...")
    
    # Stats Counters
    correct_per_slot = [0, 0, 0, 0, 0, 0] # Index 0 means 0/5 right, Index 5 means 5/5 right
    total_char_correct = 0
    total_chars = 0

    with torch.no_grad():
        for path in tqdm(image_paths):
            filename = os.path.basename(path).split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue

            # Fixed 5-way split
            h, w = img.shape
            cw = w // 5
            image_correct_count = 0
            
            # Process all 5 characters in one batch for speed
            batch_tensors = []
            for i in range(5):
                seg = img[:, i*cw:(i+1)*cw]
                seg = cv2.resize(seg, (32, 32))
                batch_tensors.append(torch.from_numpy(seg).float().unsqueeze(0) / 255.0)
            
            input_batch = torch.stack(batch_tensors).to(device)
            outputs = model(input_batch)
            predictions = torch.argmax(outputs, dim=1)

            for i in range(5):
                actual_char = filename[i] if i < len(filename) else ""
                predicted_char = ALPHABET[predictions[i].item()]
                
                if predicted_char == actual_char:
                    image_correct_count += 1
                    total_char_correct += 1
                total_chars += 1
            
            correct_per_slot[image_correct_count] += 1

    # --- FINAL REPORT ---
    print("\n" + "="*30)
    print("      ACCURACY REPORT")
    print("="*30)
    print(f"Images Processed:   {len(image_paths)}")
    print(f"Character Accuracy: {(total_char_correct/total_chars)*100:.2f}%")
    print("-" * 30)
    for i in range(6):
        pct = (correct_per_slot[i] / len(image_paths)) * 100
        print(f"{i}/5 Correct: {correct_per_slot[i]:>6} images ({pct:.1f}%)")
    
    full_wins = correct_per_slot[5]
    print("-" * 30)
    print(f"TOTAL FULLY SOLVED: {full_wins} ({(full_wins/len(image_paths))*100:.2f}%)")
    print("="*30)

if __name__ == "__main__":
    # You can set limit=None to run the whole dataset, but try 5000 first!
    run_full_evaluation(limit=20000)
    # Seed: 42