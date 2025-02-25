import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from qsimqsvd import simulate_vqsvd_with_noise
import logging
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import pandas as pd
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise import depolarizing_error, thermal_relaxation_error
from qiskit.circuit import Parameter
from qiskit import QuantumRegister, ClassicalRegister

# Set up logging configuration at the top of the file
def setup_logging():
    # Create a unique log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'qsvd_debug_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class QSVDNoiseEnv(gym.Env):
    def __init__(self, M, rank, circuit_depth=20):
        super(QSVDNoiseEnv, self).__init__()
        
        # Add matrix validation and conditioning
        print("\n=== Initializing QSVDNoiseEnv ===")
        print(f"Input matrix shape: {M.shape}")
        
        # Ensure matrix is well-conditioned
        self.M = self._condition_matrix(M)
        self.rank = rank
        self.circuit_depth = circuit_depth
        self.n = M.shape[0]
        self.n_qubits = int(np.ceil(np.log2(self.n)))
        
        # Initialize quantum backend
        self.backend = AerSimulator()
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1, high=1, 
            shape=(self.n * self.rank * 2 + self.rank,), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n * self.rank * 2 + self.rank + 1,),
            dtype=np.float32
        )
        
        # Print dimensions for debugging
        print(f"Environment dimensions:")
        print(f"Matrix size (n): {self.n}")
        print(f"Full rank: {self.rank}")
        print(f"Action space dim: {self.action_space.shape}")
        print(f"Observation space dim: {self.observation_space.shape}")
        
        # Compute reference singular values with safety checks
        self.true_s = self._compute_reference_singular_values(self.M, self.rank)
        
        # Initial state
        self.reset()

    def _condition_matrix(self, matrix):
        """Condition the input matrix to avoid numerical instability"""
        print("\n=== DEBUG: Entering _condition_matrix ===")
        print(f"DEBUG: Input matrix shape: {matrix.shape}")
        print(f"DEBUG: Input matrix type: {type(matrix)}")
        print("DEBUG: Matrix properties before conditioning:")
        print(f"DEBUG: - Is matrix None? {matrix is None}")
        print(f"DEBUG: - Contains NaN? {np.any(np.isnan(matrix))}")
        print(f"DEBUG: - Contains Inf? {np.any(np.isinf(matrix))}")
        
        # Add small regularization term to diagonal
        print("DEBUG: About to add regularization...")
        eps = 1e-8
        n = matrix.shape[0]
        print(f"DEBUG: Creating identity matrix of size {n}")
        eye_matrix = np.eye(n)
        print("DEBUG: Adding regularization term...")
        regularized = matrix + eps * eye_matrix
        print("DEBUG: Regularization complete")
        
        # Scale matrix to have reasonable norm
        print("DEBUG: About to compute Frobenius norm...")
        try:
            norm = np.linalg.norm(regularized, 'fro')
            print(f"DEBUG: Frobenius norm computed: {norm:.2e}")
            if norm > 0:
                print("DEBUG: Scaling matrix...")
                regularized = regularized / norm
                print("DEBUG: Matrix scaled successfully")
        except Exception as e:
            print(f"DEBUG: Error in norm computation/scaling: {str(e)}")
        
        # Check condition number after regularization
        print("DEBUG: About to compute condition number...")
        try:
            cond = np.linalg.cond(regularized)
            print(f"DEBUG: Matrix condition number after regularization: {cond:.2e}")
        except Exception as e:
            print(f"DEBUG: Error computing condition number: {str(e)}")
        
        return regularized

    def _compute_reference_singular_values(self, matrix, num_values, num_iterations=100):
        """Compute approximate singular values using power method with enhanced stability"""
        print("\n=== DEBUG: Entering _compute_reference_singular_values ===")
        print(f"DEBUG: Matrix shape: {matrix.shape}")
        print(f"DEBUG: Requested singular values: {num_values}")
        print("DEBUG: Matrix properties:")
        print(f"DEBUG: - Contains NaN? {np.any(np.isnan(matrix))}")
        print(f"DEBUG: - Contains Inf? {np.any(np.isinf(matrix))}")
        
        m, n = matrix.shape
        singular_values = np.zeros(num_values)
        print("DEBUG: Creating working copy of matrix...")
        A = matrix.copy()
        
        # Add small regularization
        print("DEBUG: Adding initial regularization...")
        A = A + 1e-8 * np.eye(n)
        print("DEBUG: Initial regularization complete")
        
        for i in range(num_values):
            print(f"\nDEBUG: Computing singular value {i+1}/{num_values}")
            # Initialize random vector
            print("DEBUG: Initializing random vector...")
            v = np.random.randn(n)
            print("DEBUG: About to normalize vector...")
            v = v / (np.linalg.norm(v) + 1e-12)
            print("DEBUG: Vector normalized")
            
            # Power iteration with stability checks
            print(f"DEBUG: Starting power iteration ({num_iterations * (i + 1)} iterations)")
            for iter in range(num_iterations * (i + 1)):
                if iter % 100 == 0:  # Print progress every 100 iterations
                    print(f"DEBUG: Power iteration progress: {iter}/{num_iterations * (i + 1)}")
                
                # Av
                u = A @ v
                sigma = np.linalg.norm(u)
                if sigma > 1e-12:
                    u = u / sigma
                
                # A^T u
                v = A.T @ u
                sigma = np.linalg.norm(v)
                if sigma > 1e-12:
                    v = v / sigma
            
            s_approx[i] = sigma
            v_approx[:, i] = v
            
            # Deflate
            if sigma > 1e-12:
                A = A - sigma * np.outer(u, v)
        
        print("DEBUG: Initialization complete")
        print(f"DEBUG: Initial singular values range: [{np.min(s_approx):.2e}, {np.max(s_approx):.2e}]")
        
        for r in range(self.rank):
            # Initialize vectors with stable approach
            if v_approx is not None and r < len(v_approx):
                v = v_approx[:, r].copy()
            else:
                v = np.random.randn(n)
                v = v / (np.linalg.norm(v) + 1e-12)
            
            best_sigma = 0
            best_u = None
            best_v = None
            
            # Adaptive iterations based on rank
            current_iterations = num_iterations * (1 + r // 2)
            
            for iter in range(current_iterations):
                # Create quantum circuit with improved stability
                circuit_u = self.create_quantum_matrix_circuit(v)
                
                # Parameter initialization with stable bounds
                if r < len(s_approx):
                    param_scale = min(np.pi/4, np.arcsin(min(1.0, s_approx[r])))
                    for param in circuit_u.parameters:
                        if param not in self.param_values:
                            self.param_values[param] = param_scale * np.random.uniform(0.9, 1.1)
                
                # Execute quantum circuit with error mitigation
                u = self._execute_quantum_circuit(circuit_u, use_saved_params=True)
                sigma = np.linalg.norm(u)
                
                if sigma > 1e-10:
                    u = u / sigma
                    # Stable Gram-Schmidt orthogonalization
                    if r > 0:
                        u = self._stable_orthogonalize(u, U[:, :r])
                
                circuit_v = self.create_quantum_matrix_circuit(u)
                v = self._execute_quantum_circuit(circuit_v, use_saved_params=True)
                sigma = np.linalg.norm(v)
                
                if sigma > 1e-10:
                    v = v / sigma
                    if r > 0:
                        v = self._stable_orthogonalize(v, V[:, :r])
                
                # Update best vectors if better
                if sigma > best_sigma:
                    best_sigma = sigma
                    best_u = u.copy()
                    best_v = v.copy()
            
            # Store results with proper scaling
            if best_sigma > 1e-10:
                U[:, r] = best_u
                s[r] = best_sigma * matrix_norm
                V[:, r] = best_v
                
                # Stable deflation
                deflation_matrix = best_sigma * np.outer(best_u, best_v)
                A = A - deflation_matrix
                
                # Periodic reorthogonalization
                if r > 0 and r % 2 == 0:
                    U[:, :r+1], V[:, :r+1] = self._stable_reorthogonalize(U[:, :r+1], V[:, :r+1])
            else:
                # If singular value is too small, use random initialization
                U[:, r] = np.random.randn(m)
                V[:, r] = np.random.randn(n)
                U[:, r] = U[:, r] / np.linalg.norm(U[:, r])
                V[:, r] = V[:, r] / np.linalg.norm(V[:, r])
                s[r] = 0
        
        return U, s, V, np.linalg.norm(matrix - U @ np.diag(s) @ V.T)

    def _estimate_matrix_scale(self, matrix):
        """Estimate matrix scale using quantum sampling"""
        # Sample random vectors and use quantum circuits to estimate largest magnitude
        n_samples = 10
        max_magnitude = 0
        
        for _ in range(n_samples):
            # Create random normalized vector
            v = np.random.randn(matrix.shape[1])
            v = v / np.linalg.norm(v)
            
            # Use quantum circuit to estimate Mv
            circuit = self.create_quantum_matrix_circuit(v)
            result = self._execute_quantum_circuit(circuit)
            magnitude = np.linalg.norm(result)
            max_magnitude = max(max_magnitude, magnitude)
        
        return max_magnitude

    def _quantum_sample_initial_vector(self, matrix, n):
        """Use quantum sampling to find a good initial vector"""
        # Create superposition state
        circuit = QuantumCircuit(self.n_qubits)
        circuit.h(range(self.n_qubits))  # Put all qubits in superposition
        
        # Add some random rotations to create a non-uniform superposition
        for i in range(self.n_qubits):
            angle = np.random.uniform(0, np.pi)
            circuit.ry(angle, i)
        
        # Measure
        circuit.measure_all()
        
        # Execute circuit multiple times
        counts = self._execute_with_error_mitigation(circuit)
        
        # Convert measurements to vector
        v = np.zeros(n)
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            index = int(bitstring, 2)
            if index < n:
                v[index] = np.sqrt(count / total_shots)
        
        # Normalize
        v = v / np.linalg.norm(v)
        return v

    def _ensure_orthogonality(self, U, V, current_index):
        """Ensure orthogonality of current vectors against previous ones
        
        Args:
            U: Left singular vectors
            V: Right singular vectors
            current_index: Index of the current vector
            
        Returns:
            tuple: Orthogonalized U and V matrices
        """
        # Renormalize with stability check
        norm = np.linalg.norm(U[:, current_index])
        if norm > 1e-10:
            U[:, current_index] = U[:, current_index] / norm
        else:
            U[:, current_index] = np.random.randn(U.shape[0])
            U[:, current_index] = U[:, current_index] / np.linalg.norm(U[:, current_index])
        
        norm = np.linalg.norm(V[:, current_index])
        if norm > 1e-10:
            V[:, current_index] = V[:, current_index] / norm
        else:
            V[:, current_index] = np.random.randn(V.shape[0])
            V[:, current_index] = V[:, current_index] / np.linalg.norm(V[:, current_index])
        
        return U, V

    def _stable_orthogonalize(self, vector, basis):
        """Stable orthogonalization of a vector against a basis"""
        result = vector.copy()
        for i in range(basis.shape[1]):
            # Use stable dot product
            proj = np.sum(basis[:, i] * result)
            result = result - proj * basis[:, i]
        
        # Renormalize with stability check
        norm = np.linalg.norm(result)
        if norm > 1e-10:
            result = result / norm
        else:
            result = np.random.randn(len(vector))
            result = result / np.linalg.norm(result)
        
        return result

    def _stable_reorthogonalize(self, U, V):
        """Stable reorthogonalization of U and V matrices"""
        # Use QR decomposition for stability
        U_q, U_r = np.linalg.qr(U)
        V_q, V_r = np.linalg.qr(V)
        
        # Ensure proper orientation
        signs = np.sign(np.diagonal(U_r))
        U_q = U_q * signs
        V_q = V_q * signs
        
        return U_q, V_q

    def create_quantum_matrix_circuit(self, input_vector):
        """
        Main circuit for matrix-vector multiplication
        """
        # Circuit initialization with multiple registers
        q_reg = QuantumRegister(self.n_qubits * 2, 'q')  # Double qubits for input/output
        aux_reg = QuantumRegister(1, 'aux')  # Auxiliary qubit for controlled operations
        c_reg = ClassicalRegister(2 * self.n_qubits + 1, 'c')
        circuit = QuantumCircuit(q_reg, aux_reg, c_reg)
        
        # Initialize auxiliary qubit in superposition
        circuit.h(aux_reg[0])
        
        # Encode input vector
        self._amplitude_encode_vector(circuit, input_vector, range(self.n_qubits))
        
        # Matrix multiplication operations
        for i in range(self.n_qubits):
            # Phase estimation inspired sequence
            circuit.h(q_reg[i + self.n_qubits])
            
            # Controlled operations between registers
            circuit.cx(q_reg[i], aux_reg[0])
            circuit.cx(aux_reg[0], q_reg[i + self.n_qubits])
            
            # Parameterized rotations
            for j in range(self.n_qubits):
                theta = Parameter(f'θ_{i}_{j}')
                phi = Parameter(f'φ_{i}_{j}')
                circuit.cry(theta, q_reg[j], q_reg[i + self.n_qubits])
                circuit.crz(phi, q_reg[j], q_reg[i + self.n_qubits])
        
        # Add barrier for measurement protection
        circuit.barrier()
        
        # Measure auxiliary qubit first
        circuit.measure(aux_reg[0], c_reg[0])
        
        # Measure quantum registers
        for i in range(self.n_qubits * 2):
            circuit.measure(q_reg[i], c_reg[i + 1])  # Offset by 1 for aux measurement
        
        return circuit

    def _amplitude_encode_vector(self, circuit, vector, qubits):
        """
        Quantum state preparation for classical vectors
        """
        # Normalize vector
        vector = vector / np.linalg.norm(vector)
        
        # Apply QFT for better state preparation
        self._apply_qft(circuit, qubits)
        
        # Encode amplitudes
        for i in range(len(vector)):
            if abs(vector[i]) > 1e-10:
                bin_i = format(i, f'0{len(qubits)}b')
                angle = 2 * np.arccos(abs(vector[i]))
                phase = np.angle(vector[i])
                self._multi_controlled_rotation(circuit, qubits, bin_i, angle, phase)
        
        # Apply inverse QFT
        self._apply_inverse_qft(circuit, qubits)

    def _apply_qft(self, circuit, qubits):
        """
        Quantum Fourier Transform implementation
        """
        for i in range(len(qubits)):
            circuit.h(qubits[i])
            for j in range(i+1, len(qubits)):
                phase = np.pi / float(2**(j-i))
                circuit.cp(phase, qubits[i], qubits[j])

    def _apply_inverse_qft(self, circuit, qubits):
        """
        Inverse Quantum Fourier Transform
        """
        for i in range(len(qubits)-1, -1, -1):
            for j in range(i+1, len(qubits)):
                phase = -np.pi / float(2**(j-i))
                circuit.cp(phase, qubits[i], qubits[j])
            circuit.h(qubits[i])

    def _multi_controlled_rotation(self, circuit, qubits, control_string, angle, phase):
        """Apply multi-controlled rotation based on binary control string"""
        # Convert qubits to list if it's a range
        qubits = list(qubits)
        
        # Apply X gates for 0-controls
        for i, bit in enumerate(control_string):
            if bit == '0':
                circuit.x(qubits[i])
        
        # Get control and target qubits
        control_qubits = qubits[:-1]
        target_qubit = qubits[-1]
        
        # Apply controlled rotation
        if len(control_qubits) > 0:
            circuit.mcry(angle, control_qubits, target_qubit)
            circuit.mcrz(phase, control_qubits, target_qubit)
        else:
            circuit.ry(angle, target_qubit)
            circuit.rz(phase, target_qubit)
        
        # Undo X gates
        for i, bit in enumerate(control_string):
            if bit == '0':
                circuit.x(qubits[i])

    def _execute_quantum_circuit(self, circuit, shots=1000, use_saved_params=False):
        """
        Circuit execution with error mitigation
        """
        backend = AerSimulator()
        
        # Parameter binding with bounds
        if hasattr(self, 'circuit_parameters') and circuit.parameters:
            param_dict = {}
            for param in circuit.parameters:
                if use_saved_params and param in self.param_values:
                    value = self.param_values[param]
                else:
                    value = np.random.uniform(-0.1, 0.1)
                param_dict[param] = np.clip(value, -np.pi, np.pi)
            
            circuit = circuit.assign_parameters(param_dict)
        
        # Multiple executions for error mitigation
        results = []
        weights = []
        
        for _ in range(5):
            try:
                job = backend.run(circuit, shots=shots)
                result = job.result()
                counts = result.get_counts()
                vector = self._counts_to_vector(counts)
                norm = np.linalg.norm(vector)
                
                if norm > 1e-10:
                    results.append(vector)
                    weights.append(norm)
            except Exception as e:
                continue
        
        if not results:  # If all executions failed
            return np.random.randn(2**self.n_qubits)
        
        # Weighted average with normalization
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        result = np.average(results, axis=0, weights=weights)
        
        # Ensure result is properly normalized
        norm = np.linalg.norm(result)
        if norm > 1e-10:
            result = result / norm
        
        return result

    def _counts_to_vector(self, counts):
        """Convert quantum measurement counts to vector"""
        vector = np.zeros(2**self.n_qubits)
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            # Ensure bitstring is the correct length
            bitstring = bitstring.zfill(2 * self.n_qubits + 1)  # +1 for aux qubit
            # Take only the relevant part of the bitstring for the output register
            # Skip the first bit (aux) and take the second half of remaining bits
            output_bitstring = bitstring[self.n_qubits + 1:]
            index = int(output_bitstring, 2)
            vector[index] = np.sqrt(count / total_shots)
        
        return vector

    def compute_quantum_reward(self, quantum_state, action):
        """Compute reward based on quantum measurements only"""
        # Quantum fidelity measurement
        fidelity = self._measure_quantum_fidelity(quantum_state)
        
        # Circuit complexity penalty
        depth_penalty = -0.1 * self._count_quantum_operations(action)
        
        # Quantum state purity measure
        purity = self._measure_state_purity(quantum_state)
        
        # Combined quantum-native reward
        reward = (0.5 * fidelity + 
                 0.3 * purity + 
                 0.2 * depth_penalty)
        
        return reward

    def _measure_quantum_fidelity(self, quantum_state, target_state=None):
        """Measure fidelity between current and target quantum states using SWAP test
        
        Args:
            quantum_state: Current quantum state
            target_state: Target quantum state (optional)
            
        Returns:
            float: Estimated fidelity between states
        """
        # Create SWAP test circuit
        n_qubits = self.n_qubits
        swap_circuit = QuantumCircuit(2 * n_qubits + 1, 1)  # +1 for ancilla
        
        # Prepare states
        swap_circuit.h(0)  # Ancilla in superposition
        
        # Encode states
        self._encode_vector(swap_circuit, quantum_state, range(1, n_qubits + 1))
        if target_state is not None:
            self._encode_vector(swap_circuit, target_state, range(n_qubits + 1, 2 * n_qubits + 1))
        
        # Apply SWAP test
        for i in range(n_qubits):
            swap_circuit.cswap(0, i + 1, i + n_qubits + 1)
        
        # Final Hadamard
        swap_circuit.h(0)
        swap_circuit.measure(0, 0)
        
        # Execute with error mitigation
        counts = self._execute_with_error_mitigation(swap_circuit)
        
        # Calculate fidelity from measurement statistics
        fidelity = counts.get('0', 0) / sum(counts.values())
        return 2 * fidelity - 1

    def _execute_with_error_mitigation(self, circuit, shots=1000):
        """Execute quantum circuit with error mitigation techniques
        
        Args:
            circuit (QuantumCircuit): Circuit to execute
            shots (int): Number of shots per execution
            
        Returns:
            dict: Mitigated measurement counts
        """
        backend = AerSimulator()
        
        # Richardson extrapolation for error mitigation
        scale_factors = [1, 2, 3]  # Scale noise by these factors
        results = []
        
        for scale in scale_factors:
            # Create noise-scaled circuit
            scaled_circuit = self._scale_noise(circuit, scale)
            
            # Execute circuit
            job = backend.run(scaled_circuit, shots=shots)
            counts = job.result().get_counts()
            results.append(counts)
        
        # Apply Richardson extrapolation
        mitigated_counts = self._richardson_extrapolation(results, scale_factors)
        return mitigated_counts

    def _scale_noise(self, circuit, scale_factor):
        """Scale noise in circuit by given factor
        
        Args:
            circuit (QuantumCircuit): Original circuit
            scale_factor (float): Factor to scale noise by
            
        Returns:
            QuantumCircuit: Circuit with scaled noise
        """
        # Create noise model
        noise_model = NoiseModel()
        
        # Add scaled depolarizing error
        dep_error = depolarizing_error(0.001 * scale_factor, 1)
        noise_model.add_all_qubit_quantum_error(dep_error, ['u1', 'u2', 'u3'])
        
        # Add scaled thermal relaxation
        t1, t2 = 50, 70  # microseconds
        thermal_error = thermal_relaxation_error(t1/scale_factor, t2/scale_factor, 0.1)
        noise_model.add_all_qubit_quantum_error(thermal_error, ['u1', 'u2', 'u3'])
        
        # Create noisy circuit
        noisy_circuit = circuit.copy()
        noisy_circuit.noise_model = noise_model
        
        return noisy_circuit

    def _richardson_extrapolation(self, results, scale_factors):
        """Apply Richardson extrapolation to measurement results
        
        Args:
            results (list): List of measurement counts at different noise scales
            scale_factors (list): Corresponding noise scale factors
            
        Returns:
            dict: Extrapolated (mitigated) counts
        """
        mitigated_counts = {}
        
        # Get all possible bitstrings
        bitstrings = set().union(*[res.keys() for res in results])
        
        for bitstring in bitstrings:
            # Get probabilities for each scale factor
            probs = []
            for counts in results:
                total = sum(counts.values())
                prob = counts.get(bitstring, 0) / total
                probs.append(prob)
            
            # Richardson extrapolation formula
            extrap_prob = 0
            for i, prob in enumerate(probs):
                coeff = 1
                for j, s in enumerate(scale_factors):
                    if i != j:
                        coeff *= s / (s - scale_factors[i])
                extrap_prob += coeff * prob
            
            # Convert back to counts
            if extrap_prob > 0:
                mitigated_counts[bitstring] = int(extrap_prob * 1000)  # Scale to reasonable count
        
        return mitigated_counts

    def _measure_state_purity(self, quantum_state):
        """Measure purity of quantum state using multiple SWAP tests
        
        Args:
            quantum_state: Quantum state to measure purity of
            
        Returns:
            float: Estimated purity of state
        """
        # Create circuit for purity measurement
        purity_circuit = QuantumCircuit(2 * self.n_qubits + 1, 1)
        
        # Prepare two copies of the state
        self._encode_vector(purity_circuit, quantum_state, range(self.n_qubits))
        self._encode_vector(purity_circuit, quantum_state, range(self.n_qubits, 2 * self.n_qubits))
        
        # Apply SWAP test
        purity = self._measure_quantum_fidelity(quantum_state, quantum_state)
        
        return purity

    def _encode_vector(self, circuit, vector, qubits):
        """Encode a classical vector into quantum state"""
        # Normalize vector
        vector = vector / np.linalg.norm(vector)
        
        # Convert to quantum state
        n_qubits = len(qubits)
        for i in range(len(vector)):
            if abs(vector[i]) > 1e-10:  # Only apply gates for non-zero amplitudes
                # Convert index to binary representation
                bin_i = format(i, f'0{n_qubits}b')
                # Apply X gates where needed
                for j, bit in enumerate(bin_i):
                    if bit == '1':
                        circuit.x(qubits[j])
                # Apply rotation
                if abs(vector[i]) < 1:
                    circuit.ry(2 * np.arcsin(vector[i]), qubits[0])
                # Undo X gates
                for j, bit in enumerate(bin_i):
                    if bit == '1':
                        circuit.x(qubits[j])

def add_regularization(matrix, epsilon=1e-8):
    return matrix + epsilon * np.eye(matrix.shape[0])

class MetricsTracker:
    def __init__(self, window_size=100, env=None):
        self.metrics = {
            'frobenius_error': [],
            'diag_similarity': [],
            'rewards': [],
            'orthonormality_error': [],
            'action_magnitudes': [],
            'improvement_rate': []
        }
        self.window_size = window_size
        self.rolling_metrics = {k: deque(maxlen=window_size) for k in self.metrics.keys()}
        self.env = env  # Store reference to environment

    def add_metrics(self, info, reward, action):
        # Store raw metrics
        self.metrics['frobenius_error'].append(info['frobenius_error'])
        self.metrics['diag_similarity'].append(info['diag_similarity'])
        self.metrics['rewards'].append(reward)
        self.metrics['orthonormality_error'].append(info['orthonormality_error'])
        self.metrics['action_magnitudes'].append(np.mean(np.abs(action)))
        
        # Calculate improvement rate
        if len(self.metrics['frobenius_error']) > 1:
            improvement = self.metrics['frobenius_error'][-2] - self.metrics['frobenius_error'][-1]
        else:
            improvement = 0
        self.metrics['improvement_rate'].append(improvement)

        # Update rolling metrics
        for k in self.metrics.keys():
            self.rolling_metrics[k].append(self.metrics[k][-1])

    def plot_metrics(self, save_path=None):
        """Generate and save visualization plots"""
        sns.set_style("whitegrid")
        
        # Create a 3x2 subplot figure
        fig, axes = plt.subplots(3, 2, figsize=(15, 20))
        fig.suptitle('QSVD Training Metrics', fontsize=16)

        # 1. Frobenius Error Over Time
        ax = axes[0, 0]
        ax.plot(self.metrics['frobenius_error'], label='Raw Error')
        ax.plot(pd.Series(self.metrics['frobenius_error']).rolling(20).mean(), 
                label='Moving Average', color='red')
        ax.set_title('Frobenius Error Evolution')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Error')
        ax.legend()

        # 2. Rewards Distribution
        ax = axes[0, 1]
        sns.histplot(data=self.metrics['rewards'], ax=ax, bins=30)
        ax.axvline(np.mean(self.metrics['rewards']), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.metrics['rewards']):.4f}')
        ax.set_title('Reward Distribution')
        ax.legend()

        # 3. Action Magnitudes
        ax = axes[1, 0]
        ax.plot(self.metrics['action_magnitudes'])
        ax.set_title('Average Action Magnitude')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Magnitude')

        # 4. Improvement Rate
        ax = axes[1, 1]
        ax.plot(self.metrics['improvement_rate'])
        ax.set_title('Error Improvement Rate')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Improvement')

        # 5. Diagonal Similarity
        ax = axes[2, 0]
        ax.plot(self.metrics['diag_similarity'])
        ax.set_title('Diagonal Similarity')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Similarity')

        # 6. Orthonormality Error
        ax = axes[2, 1]
        ax.plot(self.metrics['orthonormality_error'])
        ax.set_title('Orthonormality Error')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Error')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def save_metrics(self, filepath):
        """Save metrics to CSV file"""
        df = pd.DataFrame(self.metrics)
        df.to_csv(filepath, index=False)

    def plot_matrices(self, step, save_path=None):
        """Plot matrix comparisons if environment is available"""
        if self.env is None:
            return
            
        D_diag = np.diag(self.env.D_noisy)
        M_reconstructed = self.env.U_noisy @ D_diag @ self.env.V_noisy.T
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_matrix_comparison(
            self.env.M, 
            M_reconstructed,
            f'{save_path}_matrices_step_{step}.png' if save_path else None
        )
        plot_singular_values(
            self.env.M, 
            M_reconstructed,
            f'{save_path}_singvals_step_{step}.png' if save_path else None
        )

def plot_matrix_comparison(original_matrix, reconstructed_matrix, save_path=None):
    """Visualize the original and reconstructed matrices side by side"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original matrix
    im1 = ax1.imshow(original_matrix, cmap='viridis')
    ax1.set_title('Original Matrix')
    plt.colorbar(im1, ax=ax1)
    
    # Reconstructed matrix
    im2 = ax2.imshow(reconstructed_matrix, cmap='viridis')
    ax2.set_title('Reconstructed Matrix')
    plt.colorbar(im2, ax=ax2)
    
    # Difference
    difference = np.abs(original_matrix - reconstructed_matrix)
    im3 = ax3.imshow(difference, cmap='viridis')
    ax3.set_title('Absolute Difference')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_singular_values(original_matrix, reconstructed_matrix, save_path=None):
    """Compare singular value distributions"""
    # Compute singular values
    u1, s1, _ = np.linalg.svd(original_matrix)
    u2, s2, _ = np.linalg.svd(reconstructed_matrix)
    
    plt.figure(figsize=(10, 6))
    plt.plot(s1, 'b-', label='Original SVs')
    plt.plot(s2, 'r--', label='Reconstructed SVs')
    plt.yscale('log')
    plt.title('Singular Value Comparison')
    plt.xlabel('Index')
    plt.ylabel('Singular Value (log scale)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()
    
    # Create environment with explicit matrix size and rank
    matrix_size = 8
    rank = 3
    
    env = make_vec_env(
        lambda: QSVDNoiseEnv(M=np.random.rand(matrix_size, matrix_size), rank=matrix_size),
        n_envs=4
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Verify environment dimensions
    test_env = QSVDNoiseEnv(M=np.random.rand(matrix_size, matrix_size), rank=matrix_size)
    print(f"Action space shape: {test_env.action_space.shape}")
    print(f"Observation space shape: {test_env.observation_space.shape}")

    # Create PPO agent with tuned hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1
    )

    # Train for longer
    model.learn(total_timesteps=50000)

    # Modify the debug_shapes function to use logging
    def debug_shapes(logger, step_num, action, obs, rewards=None, infos=None):
        debug_info = [
            "\n=== Debug Information ===",
            f"Step number: {step_num}",
            f"Action type: {type(action)}",
            f"Action shape: {action.shape if isinstance(action, np.ndarray) else 'Not numpy array'}"
        ]
        
        if isinstance(action, np.ndarray):
            debug_info.append(f"Action content sample: {action[:2]}")
        
        debug_info.extend([
            f"Observation type: {type(obs)}",
            f"Observation shape: {obs.shape if isinstance(obs, np.ndarray) else 'Not numpy array'}"
        ])
        
        if rewards is not None:
            debug_info.extend([
                f"Rewards type: {type(rewards)}",
                f"Rewards shape: {rewards.shape if isinstance(rewards, np.ndarray) else 'Not numpy array'}",
                f"Rewards content: {rewards}"
            ])
        
        if infos is not None:
            debug_info.extend([
                f"Infos type: {type(infos)}",
                f"Infos content: {infos}"
            ])
        
        debug_info.append("=====================\n")
        logger.debug('\n'.join(debug_info))

    # Create test environment for visualization
    test_env = QSVDNoiseEnv(M=np.random.rand(matrix_size, matrix_size), rank=rank)
    metrics_tracker = MetricsTracker(env=test_env)
    
    # Evaluate
    try:
        logger.info("\nStarting evaluation...")
        obs = env.reset()[0]
        logger.info(f"Initial observation shape: {obs.shape}")
        
        total_reward = 0
        for step in range(100):
            logger.info(f"\nStep {step + 1}/100")
            
            try:
                action, _ = model.predict(obs, deterministic=True)
                debug_shapes(logger, step + 1, action, obs)
            except Exception as e:
                logger.error(f"Error during model prediction:")
                logger.error(f"Observation shape when error occurred: {obs.shape}")
                raise e
            
            try:
                if isinstance(action, np.ndarray):
                    logger.debug(f"Original action shape: {action.shape}")
                    
                    if len(action.shape) == 1:
                        action = np.tile(action, (4, 1))
                        logger.debug(f"Action shape after tiling: {action.shape}")
                    elif len(action.shape) == 2 and action.shape[0] != 4:
                        action = action.reshape(4, -1)
                        logger.debug(f"Action shape after reshaping: {action.shape}")
                    else:
                        logger.debug(f"Action shape unchanged: {action.shape}")
                    
                    logger.debug(f"Final action shape before step: {action.shape}")
                else:
                    logger.warning(f"Warning: Action is not numpy array. Type: {type(action)}")
            except Exception as e:
                logger.error("Error during action shape handling:")
                logger.error(f"Action shape when error occurred: {action.shape if isinstance(action, np.ndarray) else type(action)}")
                raise e
            
            try:
                obs, rewards, dones, infos = env.step(action)
                metrics_tracker.add_metrics(infos[0], rewards.mean(), action[0])  # Using first environment's metrics
                debug_shapes(logger, step + 1, action, obs, rewards, infos)
                
                total_reward += rewards.mean()
                logger.info(f"Step {step + 1} Results:")
                logger.info(f"- Reward: {rewards.mean():.4f}")
                logger.info(f"- Frobenius Error: {infos[0]['frobenius_error']:.4f}")
                
                if any(dones):
                    logger.info("Environment reset triggered")
                    obs = env.reset()[0]
                    logger.info(f"New observation shape after reset: {obs.shape}")
                    
            except Exception as e:
                error_info = [
                    "\n=== Error State Debug ===",
                    "Error occurred during environment step:",
                    f"- Action shape: {action.shape if isinstance(action, np.ndarray) else type(action)}",
                    f"- Observation shape: {obs.shape if isinstance(obs, np.ndarray) else type(obs)}"
                ]
                
                if 'rewards' in locals():
                    error_info.append(f"- Rewards shape: {rewards.shape if isinstance(rewards, np.ndarray) else type(rewards)}")
                if 'infos' in locals():
                    error_info.append(f"- Infos type: {type(infos)}")
                
                error_info.extend([
                    f"- Error message: {str(e)}",
                    "======================\n"
                ])
                
                logger.error('\n'.join(error_info))
                raise e

            if step % 20 == 0:  # Generate visualizations every 20 steps
                metrics_tracker.plot_matrices(step, f'qsvd_viz_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    except Exception as e:
        error_state = [
            "\n=== Final Error State ===",
            "Fatal error occurred:",
            f"- Last known action shape: {action.shape if 'action' in locals() and isinstance(action, np.ndarray) else 'Unknown'}",
            f"- Last known observation shape: {obs.shape if 'obs' in locals() and isinstance(obs, np.ndarray) else 'Unknown'}",
            f"- Error type: {type(e)}",
            f"- Error message: {str(e)}",
            "======================\n"
        ]
        logger.error('\n'.join(error_state))
        raise e

    logger.info("\nEvaluation complete!")
    logger.info(f"Average reward over 100 steps: {total_reward/100:.4f}")

    # After training/evaluation, generate and save visualizations
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_tracker.plot_metrics(f'qsvd_metrics_{timestamp}.png')
    metrics_tracker.save_metrics(f'qsvd_metrics_{timestamp}.csv')