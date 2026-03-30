import pennylane as qml
import torch
import numpy as np

# 1. Setup matching your model
n_qubits = 12
n_layers = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def circuit(inputs, weights):
    # This is how your CAPTCHA data enters the Quantum world
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # These are the "Strongly Entangling" layers (the trainable part)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Measuring the Z-axis of every qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 2. Generate dummy data to fill the "slots"
dummy_inputs = np.random.uniform(0, np.pi, n_qubits)
dummy_weights = np.random.uniform(0, np.pi, (n_layers, n_qubits, 3))

# 3. Draw the circuit
print("--- 12-QUBIT QML CIRCUIT DIAGRAM ---")
print(qml.draw_mpl(circuit)(dummy_inputs, dummy_weights))