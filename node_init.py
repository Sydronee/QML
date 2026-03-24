import pennylane as qml
from pennylane import numpy as np

# We use 2 wires, but we will manually treat them as 3-level systems
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def qudit_classifier(inputs, weights):
    # Encoding: Instead of just |0> and |1>, we use a 3x3 Unitary 
    # to move our state into a "Qutrit" space.
    for i in range(len(inputs)):
        # Custom 3x3 rotation matrix for qutrit encoding
        qml.QubitUnitary(my_qutrit_rotation(inputs[i]), wires=i)
    
    return qml.probs(wires=[0, 1])

print("Environment Ready. Qudit logic initialized.")