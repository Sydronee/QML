import os
import glob
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import pennylane as qml
from sklearn.model_selection import train_test_split
import string

# 1. Label Mapper
ALPHABET = string.digits + string.ascii_letters
CHAR_MAPPING = {char: idx for idx, char in enumerate(ALPHABET)}

def label_to_int(char):
    """Map the 62 alphanumeric characters to integer targets (0 to 61)."""
    return CHAR_MAPPING.get(char, 0)

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
n_qubits = 6
dev = qml.device("lightning.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Encode classical inputs into quantum state
    # PennyLane supports batched inputs dynamically
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # Trainable quantum layers
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    
    # Return probabilities of computational basis states
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
            nn.Linear(32, n_qubits) # Squeeze to feature vector matching qubits
        )
        
        # Quantum weights
        self.q_weights = nn.Parameter(torch.randn(n_quantum_layers, n_qubits))
        
    def forward(self, x):
        # x is the batch of images [batch_size, 1, 32, 32]
        features = self.encoder(x) # Should output [batch_size, 6]
        
        # Scale features for the quantum circuit
        features = torch.tanh(features) * torch.pi 
        
        # Process through quantum layer (batch processing)
        # Note: qml.qnode in torch typically expects a single input or has vmap support,
        # but manual stacking is safe for simple proofs of concept.
        q_out = torch.stack([quantum_circuit(f, self.q_weights) for f in features])
        
        # Slice to 62 to match A-Z, 0-9
        return q_out[:, :62]

# 4. Training Loop
def train_model(data_dir="Large_Captcha_Dataset", epochs=5, batch_size=16, lr=0.01, debug_mode=False):
    dataset = CaptchaDataset(data_dir)
    if len(dataset) == 0:
        print(f"No images found in {data_dir}. Please add some CAPTCHA images to train.")
        return
        
    # Data Splitting
    indices = list(range(len(dataset)))
    
    if debug_mode:
        print("DEBUG MODE ON: Limiting dataset to 100 samples.")
        indices = indices[:100]
        
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    model = HybridQMLModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting Training on {len(train_dataset)} training samples...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
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
            
            # Progress Feedback
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        
        # Evaluation Phase
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * test_correct / test_total if test_total > 0 else 0
        
        print(f"Epoch {epoch+1}/{epochs} Summary - Loss: {avg_loss:.4f} - Train Acc: {train_accuracy:.2f}% - Test Acc: {test_accuracy:.2f}%\n")
        
    print("Training Complete!")
    return model

if __name__ == "__main__":
    train_model()