import os
import random
import cv2
import torch
import torch.nn as nn
import pennylane as qml
import string
import numpy as np

# 1. Setup
ALPHABET = string.digits + string.ascii_letters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "Large_Captcha_Dataset"

# 2. Model Definition (Must match your QML.py)
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

# 3. Visualization Logic
def create_visual_report(num_samples=10):
    # Load Model
    model = HybridQMLModel().to(device)
    checkpoint = torch.load("qml_best.pth", map_location=device, weights_only=True)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state_dict.items()})
    model.eval()

    all_images = [f for f in os.listdir(DATA_DIR) if f.endswith(('.png', '.jpg'))]
    samples = random.sample(all_images, num_samples)
    
    canvas_list = []

    for filename in samples:
        img_path = os.path.join(DATA_DIR, filename)
        orig_img = cv2.imread(img_path)
        gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        
        # Segment (5-way split)
        h, w = gray.shape
        cw = w // 5
        pred_str = ""
        
        with torch.no_grad():
            for i in range(5):
                seg = gray[:, i*cw:(i+1)*cw]
                seg = cv2.resize(seg, (32, 32))
                tensor = torch.from_numpy(seg).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
                out = model(tensor)
                pred_str += ALPHABET[torch.argmax(out, 1).item()]

        # Draw Labels
        actual_str = filename.split('.')[0]
        display_img = cv2.resize(orig_img, (300, 100))
        cv2.putText(display_img, f"True: {actual_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        color = (0, 255, 0) if pred_str == actual_str else (0, 0, 255)
        cv2.putText(display_img, f"Pred: {pred_str}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        canvas_list.append(display_img)

    # Combine and Save
    final_gallery = np.vstack([np.hstack(canvas_list[:5]), np.hstack(canvas_list[5:])])
    cv2.imwrite("quantum_results_gallery.png", final_gallery)
    print("✅ Gallery saved as 'quantum_results_gallery.png'")

if __name__ == "__main__":
    create_visual_report()