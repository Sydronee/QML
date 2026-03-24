import os
import multiprocessing
import glob
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import pennylane as qml
from sklearn.model_selection import train_test_split
import string
from tqdm import tqdm

# Force OpenMP and PyTorch to use all CPU cores
cpu_count = os.cpu_count() or 1
os.environ["OMP_NUM_THREADS"] = str(cpu_count)
torch.set_num_threads(cpu_count)
cv2.setNumThreads(0) 

# 1. Label Mapper
ALPHABET = string.digits + string.ascii_letters
CHAR_MAPPING = {char: idx for idx, char in enumerate(ALPHABET)}

def label_to_int(char):
    return CHAR_MAPPING.get(char, 0)

# 2. Data Loader with Augmentation
class CaptchaDataset(Dataset):
    def __init__(self, data_dir="Large_Captcha_Dataset"):
        self.image_paths = glob.glob(os.path.join(data_dir, "*.png"))
        if not self.image_paths:
            self.image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
            
        # Data Augmentation to help the model generalize
        self.transform = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(5),      # Small rotations to handle slanted text
            T.ColorJitter(brightness=0.1, contrast=0.1), 
            T.ToTensor(),             # This also handles the /255.0 normalization
        ])
            
    def __len__(self):
        return len(self.image_paths) * 5 
        
    def __getitem__(self, idx):
        img_idx = idx // 5
        char_idx = idx % 5
        
        img_path = self.image_paths[img_idx]
        filename = os.path.basename(img_path).split('.')[0]
        
        char = filename[char_idx] if char_idx < len(filename) else 'A'
        label = label_to_int(char)
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return torch.zeros((1, 32, 32)), torch.tensor(label, dtype=torch.long)
            
        h, w = img.shape
        chunk_w = w // 5
        segment = img[:, char_idx * chunk_w : (char_idx + 1) * chunk_w]
        segment = cv2.resize(segment, (32, 32))
        
        # Apply transforms (includes normalization)
        segment_tensor = self.transform(segment) 
        
        return segment_tensor, torch.tensor(label, dtype=torch.long)

# 3. Hybrid Model
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
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
        
        self.q_weights = nn.Parameter(torch.empty(n_quantum_layers, n_qubits))
        torch.nn.init.uniform_(self.q_weights, a=-0.1, b=0.1)
        self.classifier = nn.Linear(n_qubits, 62)
        
    def forward(self, x):
        features = self.encoder(x)
        features = torch.tanh(features) * torch.pi 
        q_out = quantum_circuit(features, self.q_weights)
        
        if isinstance(q_out, tuple) or isinstance(q_out, list):
            q_out = torch.stack(q_out, dim=-1)
        elif q_out.ndim == 1:
            q_out = q_out.unsqueeze(0)
            
        return self.classifier(q_out.float())

# 4. Training Loop with Snapshots
def train_model(data_dir="Large_Captcha_Dataset", epochs=15, batch_size=128, lr=0.0005, debug_mode=False):
    dataset = CaptchaDataset(data_dir)
    if len(dataset) == 0:
        print(f"No images found in {data_dir}.")
        return
        
    indices = list(range(len(dataset)))
    if debug_mode:
        print("DEBUG MODE ON: Limiting dataset to 100,000 samples.")
        indices = indices[:100000]
        
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    
    num_workers = max(1, min(os.cpu_count(), 16))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, prefetch_factor=4, persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, prefetch_factor=4, persistent_workers=True, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = HybridQMLModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Scheduler: Reduces learning rate when the loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_acc = 0.0
    start_epoch = 0

    # Optional: Load checkpoint if it exists
    if os.path.exists("qml_latest.pth"):
        print("Loading existing checkpoint...")
        checkpoint = torch.load("qml_latest.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resuming from epoch {start_epoch}")

    print(f"Starting Training on {len(train_dataset)} samples...")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, labels) in enumerate(loop):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
            
        avg_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        
        # Step the scheduler
        scheduler.step(avg_loss)

        # Evaluation
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * test_correct / test_total if test_total > 0 else 0
        print(f"Epoch {epoch+1} Summary - Loss: {avg_loss:.4f} - Train: {train_accuracy:.2f}% - Test: {test_accuracy:.2f}%")

        # SAVE SNAPSHOTS
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }
        torch.save(checkpoint, "qml_latest.pth")

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(checkpoint, "qml_best.pth")
            print(f"New Best Model Saved! Accuracy: {test_accuracy:.2f}%")
        
    print("Training Complete!")
    return model

if __name__ == "__main__":
    train_model(debug_mode=True, epochs=30) 