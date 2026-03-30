import os
import multiprocessing
import glob
import cv2
import numpy as np
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
        self.n_chars = 5
            
        # Data Augmentation to help the model generalize
        self.transform = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(5),      # Small rotations to handle slanted text
            T.ColorJitter(brightness=0.1, contrast=0.1), 
            T.ToTensor(),             # This also handles the /255.0 normalization
        ])
            
    def __len__(self):
        return len(self.image_paths) * self.n_chars

    def _equal_width_segments(self, img):
        h, w = img.shape
        chunk_w = max(1, w // self.n_chars)
        segments = []
        for idx in range(self.n_chars):
            x0 = idx * chunk_w
            x1 = w if idx == self.n_chars - 1 else (idx + 1) * chunk_w
            seg = img[:, x0:x1]
            if seg.size == 0:
                seg = np.zeros((h, chunk_w), dtype=img.dtype)
            seg = cv2.resize(seg, (32, 32), interpolation=cv2.INTER_AREA)
            segments.append(seg)
        return segments

    def _contour_segments(self, img):
        h, w = img.shape
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((2, 2), dtype=np.uint8)
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = max(12, (h * w) // 500)
        min_height = max(6, int(h * 0.25))
        boxes = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw * bh < min_area or bh < min_height:
                continue
            boxes.append((x, y, bw, bh))

        if len(boxes) < self.n_chars:
            return self._equal_width_segments(img)

        if len(boxes) > self.n_chars:
            boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:self.n_chars]

        boxes = sorted(boxes, key=lambda b: b[0])

        segments = []
        for x, y, bw, bh in boxes:
            pad_x = max(2, int(0.15 * bw))
            pad_y = max(2, int(0.15 * bh))
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(w, x + bw + pad_x)
            y1 = min(h, y + bh + pad_y)

            roi = img[y0:y1, x0:x1]
            if roi.size == 0:
                return self._equal_width_segments(img)

            side = max(roi.shape)
            square = np.full((side, side), 255, dtype=np.uint8)
            oy = (side - roi.shape[0]) // 2
            ox = (side - roi.shape[1]) // 2
            square[oy:oy + roi.shape[0], ox:ox + roi.shape[1]] = roi
            segments.append(cv2.resize(square, (32, 32), interpolation=cv2.INTER_AREA))

        if len(segments) != self.n_chars:
            return self._equal_width_segments(img)

        return segments
        
    def __getitem__(self, idx):
        img_idx = idx // self.n_chars
        char_idx = idx % self.n_chars
        
        img_path = self.image_paths[img_idx]
        filename = os.path.basename(img_path).split('.')[0]
        
        char = filename[char_idx] if char_idx < len(filename) else 'A'
        label = label_to_int(char)
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return torch.zeros((1, 32, 32)), torch.tensor(label, dtype=torch.long)

        segments = self._contour_segments(img)
        segment = segments[char_idx]
        
        # Apply transforms (includes normalization)
        segment_tensor = self.transform(segment) 
        
        return segment_tensor, torch.tensor(label, dtype=torch.long)


class PreprocessedCaptchaDataset(Dataset):
    def __init__(self, processed_file="processed_data.pt", images=None, labels=None, indices=None, augment=False, noise_std=0.02):
        if images is None or labels is None:
            data = torch.load(processed_file, map_location="cpu")
            images = data.get("images")
            labels = data.get("labels")
            if images is None or labels is None:
                raise ValueError("processed_data.pt must contain 'images' and 'labels'.")

        self.images = images.float()
        self.labels = labels.long()
        self.indices = indices if indices is not None else list(range(len(self.labels)))
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image = self.images[actual_idx]
        label = self.labels[actual_idx]

        if self.augment:
            noise = torch.randn_like(image) * self.noise_std
            image = (image + noise).clamp(0.0, 1.0)

        return image, label

# 3. Hybrid Model
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
def train_model(data_dir="Large_Captcha_Dataset", processed_file="processed_data.pt", epochs=15, batch_size=256, lr=0.0002, quantum_lr=0.002, debug_mode=False, use_compile=False):
    use_preprocessed = os.path.exists(processed_file)
    if use_preprocessed:
        print(f"Using preprocessed tensor dataset: {processed_file}")
        dataset = PreprocessedCaptchaDataset(processed_file)
    else:
        print(f"Preprocessed file not found. Falling back to raw images from: {data_dir}")
        dataset = CaptchaDataset(data_dir)

    if len(dataset) == 0:
        print("No training samples found.")
        return
        
    indices = list(range(len(dataset)))
    if debug_mode:
        print("DEBUG MODE ON: Limiting dataset to 100,000 samples.")
        indices = indices[:100000]
        
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    if use_preprocessed:
        train_dataset = PreprocessedCaptchaDataset(
            images=dataset.images,
            labels=dataset.labels,
            indices=train_idx,
            augment=True,
            noise_std=0.05,
        )
        test_dataset = PreprocessedCaptchaDataset(
            images=dataset.images,
            labels=dataset.labels,
            indices=test_idx,
            augment=False,
        )
    else:
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)
    
    if use_preprocessed:
        num_workers = 0
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
    else:
        num_workers = 4
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, prefetch_factor=4, persistent_workers=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, prefetch_factor=4, persistent_workers=True, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = HybridQMLModel().to(device)
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': lr},
        {'params': [model.q_weights], 'lr': quantum_lr},
        {'params': model.classifier.parameters(), 'lr': lr},
    ], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Scheduler: Reduces learning rate when the loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_acc = 0.0
    start_epoch = 0

    def _load_checkpoint_compat(path):
        nonlocal best_acc, start_epoch
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        loaded = False
        try:
            model.load_state_dict(state_dict)
            loaded = True
        except RuntimeError:
            stripped_keys = {
                (k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k): v
                for k, v in state_dict.items()
            }
            try:
                model.load_state_dict(stripped_keys)
                loaded = True
            except RuntimeError:
                prefixed_keys = {
                    (k if k.startswith("_orig_mod.") else f"_orig_mod.{k}"): v
                    for k, v in state_dict.items()
                }
                model.load_state_dict(prefixed_keys)
                loaded = True

        if loaded and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as exc:
                print(f"Warning: optimizer state could not be loaded from {path}: {exc}")

        start_epoch = checkpoint.get('epoch', 0)
        best_acc = checkpoint.get('best_acc', best_acc)
        print(f"Resuming from epoch {start_epoch}")

    # Optional: Load checkpoint before compiling model
    if os.path.exists("qml_best.pth"):
        print("Loading best checkpoint (qml_best.pth)...")
        try:
            _load_checkpoint_compat("qml_best.pth")
        except RuntimeError as exc:
            print(f"Best checkpoint mismatch. Continuing without it. Details: {exc}")
    elif os.path.exists("qml_latest.pth"):
        print("Loading latest checkpoint (qml_latest.pth)...")
        try:
            _load_checkpoint_compat("qml_latest.pth")
        except RuntimeError as exc:
            print(f"Latest checkpoint mismatch. Continuing without it. Details: {exc}")

    # ... after loading checkpoint ...
    if os.path.exists("qml_best.pth") or os.path.exists("qml_latest.pth"):
        print("Forcing Precision Learning Rates for fine-tuning...")
        for i, group in enumerate(optimizer.param_groups):
            if i == 1: # The Quantum Weights group
                group['lr'] = 0.001
            else:      # The Encoder and Classifier groups
                group['lr'] = 0.0001

    # Compile after checkpoint loading to avoid compiled/uncompiled key mismatches
    # Disabled by default for PennyLane hybrid models because Dynamo tracing can fail.
    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"torch.compile failed, training without compile. Details: {exc}")

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
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"Current LRs: encoder={current_lrs[0]:.6f}, quantum={current_lrs[1]:.6f}, classifier={current_lrs[2]:.6f}")

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

        # --- SAVE SNAPSHOTS (Inside Loop) ---
        model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': max(test_accuracy, best_acc),
        }

        # Save 'latest' every epoch
        torch.save(checkpoint, "qml_latest.pth")

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(checkpoint, "qml_best.pth")
            print(f"✨ New Best Model Saved! Accuracy: {test_accuracy:.2f}%")

    # --- FINAL SAVE (Outside Loop) ---
    print(f"\n✅ Training Complete! Final accuracy: {test_accuracy:.2f}%")
    final_checkpoint = {
        'epoch': epochs,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }
    torch.save(final_checkpoint, "qml_final.pth")
    print("💾 Final model state saved to 'qml_final.pth'")
    return model

if __name__ == "__main__":
    train_model(debug_mode=True, epochs=95) 