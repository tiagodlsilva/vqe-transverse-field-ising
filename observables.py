from qiskit.quantum_info import SparsePauliOp, Operator
from vqe_algorithm import compute_expectation
import numpy as np

def relative_error(theo, exp):
    return np.abs(theo - exp) / np.abs(theo)

def magnetization_op(i, N_qubits):
    s = ['I'] * N_qubits; s[i] = 'Z'
    return SparsePauliOp(''.join(s), 1)

def correlation_op(i, j, N_qubits):
    s = ['I'] * N_qubits; s[i] = 'Z'; s[j] = 'Z'
    return SparsePauliOp(''.join(s), 1)

def mean_magnetization(params, circuit, N_qubits, shots=4096):
    # <M> = (1/N) Σ_i <Z_i>
    m = sum(compute_expectation(params, circuit, magnetization_op(i, N_qubits), shots=shots)
            for i in range(N_qubits))
    return m / N_qubits

def correlation(params, circuit, i, r, N_qubits, shots=4096):
    # Two-point function <Z_i Z_{i+r}> with periodic boundaries
    j = (i + r) % N_qubits
    return compute_expectation(params, circuit, correlation_op(i, j, N_qubits), shots=shots)


def mean_correlation(params, circuit, r, N_qubits, shots = 4096):
    mean = sum(correlation(params, circuit, i, r, N_qubits, shots=shots) for i in range(N_qubits))
    return mean / N_qubits