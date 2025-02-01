import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, pauli_error, depolarizing_error
from scipy.linalg import svd as classical_svd
from qiskit.quantum_info.operators import Pauli
from qiskit.circuit.library import IGate, XGate, YGate, ZGate


def simulate_vqsvd_with_noise(M, rank, circuit_depth=20):
    """
    Simulate quantum-inspired SVD with noise
    
    Parameters:
    -----------
    M : ndarray
        Input matrix of shape (n, n)
    rank : int
        Desired rank of approximation (used for noise model)
    circuit_depth : int
        Depth of simulated quantum circuit (affects noise level)
    """
    n = M.shape[0]
    full_rank = min(M.shape)  # Use full rank for computation
    
    def power_method(matrix, num_iterations=100):
        m, n = matrix.shape
        U = np.zeros((m, full_rank))
        s = np.zeros(full_rank)
        V = np.zeros((n, full_rank))
        
        A = matrix.copy()
        for r in range(full_rank):
            # Initialize random vector
            v = np.random.randn(n)
            v = v / np.linalg.norm(v)
            
            # Power iteration with increased iterations for smaller singular values
            num_iter = num_iterations * (1 + r)  # More iterations for smaller values
            for _ in range(num_iter):
                # Compute left singular vector
                u = A @ v
                sigma = np.linalg.norm(u)
                if sigma > 1e-10:
                    u = u / sigma
                
                # Compute right singular vector
                v = A.T @ u
                sigma = np.linalg.norm(v)
                if sigma > 1e-10:
                    v = v / sigma
            
            # Store the computed vectors and value
            U[:, r] = u
            s[r] = sigma
            V[:, r] = v
            
            # Deflate the matrix with increased numerical stability
            A = A - sigma * np.outer(u, v)
            # Reorthogonalize against previous vectors
            if r > 0:
                U[:, r] = U[:, r] - U[:, :r] @ (U[:, :r].T @ U[:, r])
                V[:, r] = V[:, r] - V[:, :r] @ (V[:, :r].T @ V[:, r])
                U[:, r] = U[:, r] / np.linalg.norm(U[:, r])
                V[:, r] = V[:, r] / np.linalg.norm(V[:, r])
        
        return U, s, V
    
    # Compute approximate SVD using power method
    U_true, s_true, V_true = power_method(M)
    
    # Generate noise that decreases more gradually for smaller singular values
    noise_scale = 1.0 / (1.0 + np.arange(full_rank))  # Hyperbolic decay instead of exponential
    base_noise_level = 0.05  # Reduced base noise level
    singular_value_noise = np.random.normal(0, base_noise_level * noise_scale)
    approx_singular_values = s_true * (1 + singular_value_noise)
    
    # Add noise to singular vectors with more gradual decay
    noise_level = 0.1 / np.sqrt(circuit_depth)  # Square root scaling for better stability
    vector_noise_scale = 1.0 / np.sqrt(1.0 + np.arange(full_rank))  # More gradual decay
    U_noise = np.random.randn(*U_true.shape) * noise_level * vector_noise_scale[None, :]
    V_noise = np.random.randn(*V_true.shape) * noise_level * vector_noise_scale[None, :]
    
    U_noisy = U_true + U_noise
    V_noisy = V_true + V_noise
    
    # Ensure orthonormality through modified Gram-Schmidt process
    def gram_schmidt(matrix):
        Q = np.zeros_like(matrix)
        for i in range(matrix.shape[1]):
            q = matrix[:, i]
            # Double Gram-Schmidt for better numerical stability
            for _ in range(2):
                for j in range(i):
                    q = q - np.dot(Q[:, j], q) * Q[:, j]
            norm = np.linalg.norm(q)
            if norm > 1e-10:
                q = q / norm
            Q[:, i] = q
        return Q
    
    U_noisy = gram_schmidt(U_noisy)
    V_noisy = gram_schmidt(V_noisy)
    
    # Create diagonal matrix of correct shape
    D = np.zeros((full_rank, full_rank))
    np.fill_diagonal(D, approx_singular_values)
    
    # Reconstruct matrix
    M_reconstructed = U_noisy @ D @ V_noisy.T
    
    # Compute error
    frobenius_error = np.linalg.norm(M - M_reconstructed, ord='fro') / np.linalg.norm(M, ord='fro')
    
    # Print shapes for debugging
    print(f"Shapes - U_noisy: {U_noisy.shape}, D: {D.shape}, V_noisy: {V_noisy.shape}")
    print(f"Original matrix shape: {M.shape}")
    print(f"Reconstructed matrix shape: {M_reconstructed.shape}")
    
    return U_noisy, approx_singular_values, V_noisy, frobenius_error
