import os
import glob
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pennylane as qml

# 1. Label Mapper
def label_to_int(char):
    """Map the 4 most common characters to quantum states (0 to 3)."""
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    # Fallback to 0 if character is not A, B, C, or D for proof of concept
    return mapping.get(char.upper(), 0)

# 2. Data Loader
class CaptchaDataset(Dataset):
    def __init__(self, data_dir="Large_Captcha_Dataset"):
        self.image_paths = glob.glob(os.path.join(data_dir, "*.png"))
        if not self.image_paths:
            self.image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
            
    def __len__(self):
        return len(self.image_paths) * 5 # 5 segments per image
        
    def __getitem__(self, idx):
        img_idx = idx // 5
        char_idx = idx % 5
        
        img_path = self.image_paths[img_idx]
        filename = os.path.basename(img_path).split('.')[0]
        
        # Parse corresponding character
        char = filename[char_idx] if char_idx < len(filename) else 'A'
        label = label_to_int(char)
        
        # Load and grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return torch.zeros((1, 32, 32)), torch.tensor(label, dtype=torch.long)
            
        # Resize to (160, 32) so it can be sliced evenly into 5 parts width-wise
        img = cv2.resize(img, (160, 32))
        
        # Slice segment (32x32)
        segment = img[:, char_idx*32:(char_idx+1)*32]
        
        # Normalize and convert to tensor
        segment = segment.astype("float32") / 255.0
        segment_tensor = torch.tensor(segment).unsqueeze(0) # (1, 32, 32)
        
        return segment_tensor, torch.tensor(label, dtype=torch.long)

# 3. Classical Encoder & Quantum Layer
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Encode classical inputs into quantum state
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # Trainable quantum layers
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    
    # Return probabilities of computational basis states (|00>, |01>, |10>, |11>)
    return qml.probs(wires=range(n_qubits))

class HybridQMLModel(nn.Module):
    def __init__(self, n_quantum_layers=2):
        super().__init__()
        # Classical CNN Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8x8
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # Squeeze to 2-element feature vector
        )
        
        # Quantum weights
        self.q_weights = nn.Parameter(torch.randn(n_quantum_layers, n_qubits))
        
    def forward(self, x):
        # Classical forward
        features = self.encoder(x)
        features = torch.tanh(features) * torch.pi # Scale for AngleEmbedding
        
        # Process through quantum layer (batch processing)
        # Note: qml.qnode in torch typically expects a single input or has vmap support,
        # but manual stacking is safe for simple proofs of concept.
        q_out = torch.stack([quantum_circuit(f, self.q_weights) for f in features])
        
        return q_out

# 4. Training Loop
def train_model(data_dir="Large_Captcha_Dataset", epochs=5, batch_size=16, lr=0.01):
    dataset = CaptchaDataset(data_dir)
    if len(dataset) == 0:
        print(f"No images found in {data_dir}. Please add some CAPTCHA images to train.")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = HybridQMLModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    print("Starting Training...")
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")
        
    print("Training Complete!")
    return model

if __name__ == "__main__":
    train_model()
