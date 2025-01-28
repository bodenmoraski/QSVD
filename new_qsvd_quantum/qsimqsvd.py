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
    print("\n=== DEBUG: Entering simulate_vqsvd_with_noise ===")
    print(f"DEBUG: Initial matrix shape: {M.shape}")
    print(f"DEBUG: Initial matrix type: {type(M)}")
    print("DEBUG: About to compute condition number...")
    
    n = M.shape[0]
    full_rank = min(M.shape)
    
    print(f"\n=== DEBUG: Starting VQSVD Simulation ===")
    print(f"DEBUG: Matrix shape: {M.shape}")
    print("DEBUG: About to compute condition number...")
    
    # Suppress warnings for condition number calculations
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            cond = np.linalg.cond(M)
            print(f"DEBUG: Condition number computed successfully: {cond:.2e}")
        except Exception as e:
            print(f"DEBUG: Error computing condition number: {str(e)}")
    
    def power_method(matrix, num_iterations=100):
        """Compute approximate singular values using power method with enhanced stability"""
        print("\n=== DEBUG: Entering power_method ===")
        print(f"DEBUG: Matrix input shape: {matrix.shape}")
        print(f"DEBUG: Matrix input type: {type(matrix)}")
        print("DEBUG: Matrix properties before any operations:")
        print(f"DEBUG: - Is matrix None? {matrix is None}")
        print(f"DEBUG: - Contains NaN? {np.any(np.isnan(matrix))}")
        print(f"DEBUG: - Contains Inf? {np.any(np.isinf(matrix))}")
        
        # Print matrix properties before regularization
        print(f"\nDEBUG: Power Method - Initial matrix properties:")
        print(f"DEBUG: - Matrix shape: {matrix.shape}")
        print("DEBUG: About to compute Frobenius norm...")
        
        # Suppress warnings for condition number calculations
        with np.errstate(divide='ignore', invalid='ignore'):
            try:
                norm = np.linalg.norm(matrix, 'fro')
                print(f"DEBUG: - Matrix norm computed: {norm:.2e}")
                
                print("DEBUG: About to compute condition number...")
                cond = np.linalg.cond(matrix)
                print(f"DEBUG: - Initial condition number computed: {cond:.2e}")
            except Exception as e:
                print(f"DEBUG: Error in computations: {str(e)}")
        
        print("DEBUG: About to apply regularization...")
        # More aggressive regularization
        reg_factor = 1e-6
        n = matrix.shape[0]
        print(f"DEBUG: Creating identity matrix of size {n}")
        eye_matrix = np.eye(n)
        print("DEBUG: About to add regularization term...")
        matrix = matrix + reg_factor * eye_matrix
        print("DEBUG: Regularization applied successfully")
        
        # Scale matrix
        print("DEBUG: About to scale matrix...")
        matrix_norm = np.linalg.norm(matrix, 'fro')
        if matrix_norm > 1e-10:
            matrix = matrix / matrix_norm
            print(f"DEBUG: Matrix scaled by {matrix_norm:.2e}")
        else:
            print("DEBUG: Matrix norm too small for scaling")
        
        print("DEBUG: About to compute post-regularization condition number...")
        try:
            cond_post = np.linalg.cond(matrix)
            print(f"DEBUG: - After regularization - condition number: {cond_post:.2e}")
        except Exception as e:
            print(f"DEBUG: Error computing post-regularization condition number: {str(e)}")
        
        m, n = matrix.shape
        full_rank = min(matrix.shape)
        U = np.zeros((m, full_rank))
        s = np.zeros(full_rank)
        V = np.zeros((n, full_rank))
        
        A = matrix.copy()
        for r in range(full_rank):
            if r % 5 == 0:
                print(f"\nComputing singular triplet {r}/{full_rank}")
                try:
                    current_norm = np.linalg.norm(A, 'fro')
                    print(f"- Current matrix norm: {current_norm:.2e}")
                except:
                    print("- Could not compute matrix norm")
            
            # Initialize random vector with explicit normalization
            v = np.random.randn(n)
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-10:
                v = v / norm_v
            else:
                v = np.random.randn(n)
                v = v / np.linalg.norm(v)
            
            # Power iteration with enhanced stability
            num_iter = num_iterations * (1 + r)
            for _ in range(num_iter):
                try:
                    # Compute left singular vector with stability checks
                    u = A @ v
                    sigma = np.linalg.norm(u)
                    if sigma > 1e-10:
                        u = u / sigma
                    else:
                        # Reinitialize if sigma is too small
                        u = np.random.randn(m)
                        u = u / np.linalg.norm(u)
                        break
                    
                    # Compute right singular vector
                    v = A.T @ u
                    sigma = np.linalg.norm(v)
                    if sigma > 1e-10:
                        v = v / sigma
                    else:
                        # Reinitialize if sigma is too small
                        v = np.random.randn(n)
                        v = v / np.linalg.norm(v)
                        break
                except np.linalg.LinAlgError as e:
                    print(f"DEBUG: LinAlgError during power iteration: {e}")
                    break  # Exit iteration on error
            
            # Store results with explicit normalization
            try:
                U[:, r] = u / np.linalg.norm(u)
                s[r] = sigma * matrix_norm  # Scale back to original matrix size
                V[:, r] = v / np.linalg.norm(v)
            except ZeroDivisionError as e:
                print(f"DEBUG: ZeroDivisionError during normalization: {e}")
                U[:, r] = u
                V[:, r] = v
                s[r] = sigma
            
            # Safe deflation with stability check
            deflation = sigma * np.outer(u, v)
            if not np.any(np.isnan(deflation)):
                A = A - deflation
                
                # Add small regularization after deflation
                A = A + (reg_factor * 1e-2) * np.eye(n)
            else:
                print(f"Warning: Skipping unstable deflation at rank {r}")
            
            # Reorthogonalization with stability
            if r > 0:
                # Orthogonalize U
                for j in range(r):
                    proj = np.dot(U[:, j], U[:, r])
                    U[:, r] = U[:, r] - proj * U[:, j]
                norm_u = np.linalg.norm(U[:, r])
                if norm_u > 1e-10:
                    U[:, r] = U[:, r] / norm_u
                
                # Orthogonalize V
                for j in range(r):
                    proj = np.dot(V[:, j], V[:, r])
                    V[:, r] = V[:, r] - proj * V[:, j]
                norm_v = np.linalg.norm(V[:, r])
                if norm_v > 1e-10:
                    V[:, r] = V[:, r] / norm_v
                else:
                    # Reinitialize if norm is too small
                    V[:, r] = np.random.randn(n)
                    V[:, r] = V[:, r] / np.linalg.norm(V[:, r])
        
        return U, s, V
    
    # Compute approximate SVD
    try:
        print("\nStarting power method computation...")
        U_true, s_true, V_true = power_method(M)
        print("\nPower method completed successfully")
        print(f"Singular values range: [{s_true.min():.2e}, {s_true.max():.2e}]")
    except np.linalg.LinAlgError as e:
        print(f"\nError in power method: {e}")
        return None, None, None, None
    
    # Generate noise
    noise_scale = np.exp(-np.arange(full_rank) / rank)
    singular_value_noise = np.random.normal(0, 0.1 * noise_scale)
    approx_singular_values = s_true * (1 + singular_value_noise)
    
    # Add noise to singular vectors
    noise_level = 1 / circuit_depth
    U_noise = np.random.randn(*U_true.shape) * noise_level * noise_scale[None, :]
    V_noise = np.random.randn(*V_true.shape) * noise_level * noise_scale[None, :]
    
    U_noisy = U_true + U_noise
    V_noisy = V_true + V_noise
    
    # Orthonormalization
    def gram_schmidt(matrix):
        print(f"\nPerforming Gram-Schmidt - Matrix shape: {matrix.shape}")
        Q = np.zeros_like(matrix)
        for i in range(matrix.shape[1]):
            q = matrix[:, i]
            for _ in range(2):  # Double Gram-Schmidt
                for j in range(i):
                    q = q - np.dot(Q[:, j], q) * Q[:, j]
            norm = np.linalg.norm(q)
            if norm < 1e-10:
                print(f"Warning: Small norm ({norm:.2e}) at column {i}")
            q = q / (norm + 1e-12)  # Add small constant for stability
            Q[:, i] = q
        return Q
    
    U_noisy = gram_schmidt(U_noisy)
    V_noisy = gram_schmidt(V_noisy)
    
    # Create diagonal matrix
    D = np.zeros((full_rank, full_rank))
    np.fill_diagonal(D, approx_singular_values)
    
    # Reconstruct matrix
    M_reconstructed = U_noisy @ D @ V_noisy.T
    
    # Compute error
    try:
        frobenius_error = np.linalg.norm(M - M_reconstructed, ord='fro') / np.linalg.norm(M, ord='fro')
        print(f"\nFinal Frobenius error: {frobenius_error:.2e}")
    except np.linalg.LinAlgError as e:
        print(f"\nError computing Frobenius norm: {e}")
        frobenius_error = float('inf')
    
    return U_noisy, approx_singular_values, V_noisy, frobenius_error
