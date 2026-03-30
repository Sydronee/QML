import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import string
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 1. Setup & Alphabet
ALPHABET = string.digits + string.ascii_letters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROCESSED_FILE = "processed_data.pt"

# 2. Preprocessed Data Loader (Matches your QML setup exactly)
class PreprocessedDataset(Dataset):
    def __init__(self, processed_file, indices, augment=False):
        data = torch.load(processed_file, map_location="cpu")
        self.images = data["images"][indices].float()
        self.labels = data["labels"][indices].long()
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.augment:
            # Subtle noise to match the QML training conditions
            noise = torch.randn_like(image) * 0.05
            image = (image + noise).clamp(0.0, 1.0)
        return image, label

# 3. The Classical Model
class ClassicalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # EXACT same encoder as your HybridQMLModel
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2), 
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), 
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU()
        )
        # Instead of 12 qubits, we use a wide classical "Head"
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # Standard regularization
            nn.Linear(128, 62)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)

# 4. Training Engine
def run_classical_training(epochs=100, batch_size=256):
    print(f"Loading {PROCESSED_FILE}...")
    full_data = torch.load(PROCESSED_FILE, map_location="cpu")
    indices = list(range(len(full_data["labels"])))
    
    # Using the same 100k limit as your QML debug mode for a fair fight
    indices = indices[:100000] 
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_loader = DataLoader(PreprocessedDataset(PROCESSED_FILE, train_idx, augment=True), 
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(PreprocessedDataset(PROCESSED_FILE, test_idx), 
                             batch_size=batch_size)

    model = ClassicalCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"🚀 Training Classical Model on {device}...")
    
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for imgs, lbls in loop:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            _, pred = torch.max(outputs, 1)
            total += lbls.size(0)
            correct += (pred == lbls).sum().item()
            loop.set_postfix(acc=100.*correct/total)

        # Quick Test Eval
        model.eval()
        t_correct, t_total = 0, 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                _, pred = torch.max(outputs, 1)
                t_total += lbls.size(0)
                t_correct += (pred == lbls).sum().item()
        
        print(f"Epoch {epoch+1} Test Acc: {100.*t_correct/t_total:.2f}%")

    torch.save(model.state_dict(), "classical_best.pth")
    print("✅ Classical Model Saved as 'classical_best.pth'")

if __name__ == "__main__":
    run_classical_training()