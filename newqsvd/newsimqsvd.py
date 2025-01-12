import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, pauli_error, depolarizing_error
from scipy.linalg import svd as classical_svd
from qiskit.quantum_info.operators import Pauli
from qiskit.circuit.library import IGate, XGate, YGate, ZGate


def simulate_vqsvd_with_noise(M, rank, circuit_depth=20):
    n = M.shape[0]
    
    # Generate singular values with noise
    true_singular_values = np.sort(np.random.rand(rank))[::-1]
    singular_value_noise = np.random.normal(0, 0.1, size=rank)  # 10% noise
    approx_singular_values = true_singular_values * (1 + singular_value_noise)
    
    # Generate singular vectors with noise
    U = np.linalg.qr(np.random.randn(n, rank))[0]
    V = np.linalg.qr(np.random.randn(n, rank))[0]
    noise_level = 1 / circuit_depth
    U_noisy = U + noise_level * np.random.randn(*U.shape)
    V_noisy = V + noise_level * np.random.randn(*V.shape)
    
    # Ensure orthonormality
    U_noisy, _ = np.linalg.qr(U_noisy)
    V_noisy, _ = np.linalg.qr(V_noisy)
    
    # Construct reconstructed matrix using rank x rank diagonal matrix
    D = np.diag(approx_singular_values)  # This creates a rank x rank diagonal matrix
    M_reconstructed = U_noisy @ D @ V_noisy.T
    
    # Compute error
    frobenius_error = np.linalg.norm(M - M_reconstructed, ord='fro') / np.linalg.norm(M, ord='fro')
    
    return U_noisy, approx_singular_values, V_noisy, frobenius_error
