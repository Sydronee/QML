# Hybrid Quantum Machine Learning for CAPTCHA Recognition

This repository contains a deep learning project that explores the use of **Hybrid Quantum Machine Learning (QML)** to decode and recognize alphanumeric CAPTCHAs. Built primarily using **PyTorch** and **PennyLane**, combining classical feature extraction and readout layers with a Parameterized Quantum Circuit (PQC). 

A baseline Classical Convolutional Neural Network (CNN) is also provided to benchmark the performance and parameter efficiency of the quantum approach.

## 🚀 Features

- **Hybrid Quantum-Classical Architecture**: Integrates PyTorch neural network layers with PennyLane quantum circuits (`qml.AngleEmbedding`, `qml.StronglyEntanglingLayers`).
- **Parallel Processing Setup**: Multi-core enabled PyTorch and OpenCV dataloaders for efficient image processing.
- **Classical CNN Baseline**: Fair comparison against strictly classical CNN models.
- **Circuit Visualization**: Tools to draw and generate layouts of the quantum circuits being simulated.
- **Inference & Visualization Tools**: Easy-to-use scripts for evaluating models on random CAPTCHA images and generating gallery results.

## 📂 Repository Structure

- `QML.py` : Main training script for the Hybrid QML model.
- `classical_cnn.py` : Training script for the pure classical CNN baseline.
- `interferenceScript.py` : Inference script to test a trained QML model on new/unseen CAPTCHA images.
- `visualize_results.py` : Randomly selects CAPTCHAs, runs inference using the QML model, and visualizes the predictions against truths.
- `draw_circuit.py` : Generates visualizations (like `quantum_gates_unrolled.png` and PDF diagrams) of the quantum circuit architecture.
- `eval_stats.py` : Evaluates model statistics, calculates dataset accuracies, and benchmarks QML performance.
- `qml.ipynb` : Jupyter Notebook for interactive experiments and step-by-step QML training.
- `requirements.txt` : List of Python dependencies.

### Key Artifacts (Git-ignored)
- `Large_Captcha_Dataset/` : The dataset of ground-truth alphanumeric CAPTCHAs.
- `processed_data.pt` : Pre-processed/tensorized CAPTCHA dataset.
- `*.pth` files: PyTorch model weights (e.g., `qml_best.pth`, `classical_best.pth`).

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd CAT2
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv qudit_env
   source qudit_env/bin/activate  # On Windows: qudit_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Data:**
   Extract your CAPTCHA dataset into a folder named `Large_Captcha_Dataset/` at the root of the project. (If you have it as `archive.zip`, unzip it).

## 🧠 Model Architecture

The quantum model (`HybridQMLModel`) utilizes **12 Qubits** simulated on the `default.qubit` device. 
1. **Classical Downscaling**: A PyTorch Dense/Linear layer maps image features down to 12 dimensions.
2. **Quantum Layer**: The 12 features are encoded as angles via `qml.AngleEmbedding`. A trainable quantum layer (`qml.StronglyEntanglingLayers`) operates on these qubits.
3. **Measurement**: We measure the Pauli-Z expectation value resulting in 12 continuous values.
4. **Classical Readout**: The values are passed through a final multi-layer perceptron to output character predictions.

## ⚙️ Usage

### Training the QML Model
Run the main script to start training the hybrid model. Ensure the dataset is correctly placed.
```bash
python QML.py
```
This saves checkpoints like `qml_best.pth` and `qml_latest.pth`.

### Training the Classical Baseline
To compare with a standard convolutional network:
```bash
python classical_cnn.py
```
Weights will be saved as `classical_best.pth`.

### Running Inference
To predict a specific CAPTCHA using the trained hybrid model:
```bash
python interferenceScript.py path/to/captcha.png
```

### Evaluating and Visualizing
To compute accuracy metrics over the testing set:
```bash
python eval_stats.py
```

To visualize predictions in a gallery window:
```bash
python visualize_results.py
```

### Visualizing the Quantum Circuit
Generate a PDF and PNG visual representation of your QML setup:
```bash
python draw_circuit.py
```

## 📝 Requirements

Major libraries used:
- `torch`, `torchvision` (PyTorch)
- `pennylane` (Quantum Machine Learning)
- `opencv-python` (`cv2`)
- `scikit-learn`
- `numpy`, `matplotlib`

Please refer to `requirements.txt` for exact versions.
