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
        self.M = M
        self.rank = rank
        self.circuit_depth = circuit_depth
        self.n = M.shape[0]
        self.n_qubits = int(np.ceil(np.log2(self.n)))  # Number of qubits needed
        
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
        
        # Compute reference singular values
        self.true_s = self._compute_reference_singular_values(M, self.rank)
        
        # Initial state
        self.reset()

    def _compute_reference_singular_values(self, matrix, num_values, num_iterations=100):
        """Compute approximate singular values using power method"""
        m, n = matrix.shape
        singular_values = np.zeros(num_values)
        A = matrix.copy()
        
        for i in range(num_values):
            # Initialize random vector
            v = np.random.randn(n)
            v = v / np.linalg.norm(v)
            
            # Power iteration
            for _ in range(num_iterations * (i + 1)):  # More iterations for smaller values
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
            
            # Store singular value
            singular_values[i] = sigma
            
            # Deflate matrix
            A = A - sigma * np.outer(u, v)
        
        return singular_values

    def _quantum_deflation(self, matrix, singular_value, left_vector, right_vector):
        """Basic quantum deflation process
        
        Args:
            matrix: Current matrix
            singular_value: Found singular value
            left_vector: Left singular vector
            right_vector: Right singular vector
            
        Returns:
            ndarray: Deflated matrix
        """
        # Create deflation circuit
        q_reg = QuantumRegister(self.n_qubits * 2, 'q')  # Double qubits for left and right vectors
        c_reg = ClassicalRegister(self.n_qubits * 2, 'c')  # Classical register for measurements
        circuit = QuantumCircuit(q_reg, c_reg)
        
        # Encode vectors
        self._encode_vector(circuit, left_vector, range(self.n_qubits))
        self._encode_vector(circuit, right_vector, range(self.n_qubits, 2*self.n_qubits))
        
        # Apply controlled operations for deflation
        for i in range(self.n_qubits):
            circuit.cx(q_reg[i], q_reg[i + self.n_qubits])
        
        # Add measurements for all qubits
        circuit.measure(q_reg, c_reg)
        
        # Execute circuit with error mitigation
        backend = AerSimulator()
        job = backend.run(circuit, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Process measurement results
        measured_state = self._counts_to_vector(counts)
        deflated = matrix - singular_value * np.outer(left_vector, right_vector)
        return deflated

    def reset(self, seed=None, options=None):
        """Reset environment with quantum initialization"""
        super().reset(seed=seed)
        
        # Initialize quantum state
        self.U_noisy, self.D_noisy, self.V_noisy, _ = self.quantum_power_iteration(self.M)
        
        # Create observation
        observation = np.concatenate([
            self.U_noisy.flatten(),
            self.D_noisy,
            self.V_noisy.flatten(),
            [0.0]  # Initial error
        ])
        
        print(f"Reset - Shapes:")
        print(f"U_noisy: {self.U_noisy.shape}")
        print(f"D_noisy: {self.D_noisy.shape}")
        print(f"V_noisy: {self.V_noisy.shape}")
        print(f"Observation: {observation.shape}")
        
        return observation, {}

    def step(self, action):
        # Reshape action to match matrices
        action = np.array(action, dtype=np.float32).flatten()
        
        # Split action into U, D, and V adjustments
        u_end = self.n * self.rank
        d_end = u_end + self.rank
        
        U_adjust = action[:u_end].reshape(self.n, self.rank)
        D_adjust = action[u_end:d_end]
        V_adjust = action[d_end:].reshape(self.n, self.rank)
        
        # Apply adjustments
        self.U_noisy += U_adjust
        self.D_noisy += D_adjust
        self.V_noisy += V_adjust
        
        # Ensure orthonormality
        self.U_noisy, _ = np.linalg.qr(self.U_noisy)
        self.V_noisy, _ = np.linalg.qr(self.V_noisy)
        
        # Reconstruct matrix
        M_reconstructed = self.U_noisy @ np.diag(self.D_noisy) @ self.V_noisy.T
        
        # Calculate error and reward
        frobenius_error = np.linalg.norm(self.M - M_reconstructed, ord='fro') / np.linalg.norm(self.M, ord='fro')
        reward = -frobenius_error
        
        # Calculate diagonal similarity using pre-computed reference values
        diag_similarity = 1.0 - np.mean(np.abs(self.true_s - self.D_noisy) / (self.true_s + 1e-8))
        
        # Orthonormality error
        u_ortho_error = np.mean(np.abs(self.U_noisy.T @ self.U_noisy - np.eye(self.rank)))
        v_ortho_error = np.mean(np.abs(self.V_noisy.T @ self.V_noisy - np.eye(self.rank)))
        orthonormality_error = (u_ortho_error + v_ortho_error) / 2
        
        # Create observation
        observation = np.concatenate([
            self.U_noisy.flatten(),
            self.D_noisy,
            self.V_noisy.flatten(),
            [frobenius_error]
        ])
        
        done = False
        info = {
            'frobenius_error': frobenius_error,
            'diag_similarity': diag_similarity,
            'orthonormality_error': orthonormality_error
        }
        
        return observation, reward, done, False, info

    def compute_reward(self, M_reconstructed):
        # Add immediate rewards for small improvements
        current_error = np.linalg.norm(self.M - M_reconstructed, ord='fro') / np.linalg.norm(self.M, ord='fro')
        if hasattr(self, 'last_error'):
            improvement = self.last_error - current_error
            improvement_reward = np.clip(improvement * 20, -1, 1)  # Increased weight
        else:
            improvement_reward = 0
        self.last_error = current_error
        
        # Penalize extreme actions
        action_penalty = -0.1 * np.mean(np.abs(action))  # New penalty
        
        return improvement_reward + action_penalty

    def quantum_power_iteration(self, matrix, num_iterations=100):
        """Quantum version of power iteration"""
        m, n = matrix.shape
        U = np.zeros((m, self.rank))
        s = np.zeros(self.rank)
        V = np.zeros((n, self.rank))
        
        A = matrix.copy()
        for r in range(self.rank):
            # Initialize random vector
            v = np.random.randn(n)
            v = v / np.linalg.norm(v)
            
            for _ in range(num_iterations):
                # Create and execute quantum circuits
                circuit_u = self.create_quantum_matrix_circuit(v)
                u = self._execute_quantum_circuit(circuit_u)
                sigma = np.linalg.norm(u)
                if sigma > 1e-10:
                    u = u / sigma
                
                circuit_v = self.create_quantum_matrix_circuit(u)
                v = self._execute_quantum_circuit(circuit_v)
                sigma = np.linalg.norm(v)
                if sigma > 1e-10:
                    v = v / sigma
            
            U[:, r] = u
            s[r] = sigma
            V[:, r] = v
            
            # Quantum deflation
            A = self._quantum_deflation(A, sigma, u, v)
        
        return U, s, V, np.linalg.norm(matrix - U @ np.diag(s) @ V.T)

    def create_quantum_matrix_circuit(self, input_vector):
        """Create quantum circuit for matrix-vector multiplication
        
        Args:
            input_vector (ndarray): Input vector for multiplication
            
        Returns:
            QuantumCircuit: Circuit implementing matrix multiplication
        """
        # Create circuit with quantum and classical registers
        q_reg = QuantumRegister(self.n_qubits * 2, 'q')
        c_reg = ClassicalRegister(self.n_qubits * 2, 'c')  # Measure all qubits
        circuit = QuantumCircuit(q_reg, c_reg)
        
        # Encode input vector
        self._encode_vector(circuit, input_vector, range(self.n_qubits))
        
        # Store parameters for later binding
        self.circuit_parameters = []
        
        # Add quantum operations for matrix multiplication
        for i in range(self.n_qubits):
            # Add controlled operations between input and output registers
            circuit.cx(q_reg[i], q_reg[i + self.n_qubits])
            
            # Add parameterized rotations
            theta = Parameter(f'θ_{i}')
            self.circuit_parameters.append(theta)
            circuit.ry(theta, q_reg[i + self.n_qubits])
            
            # Add phase operations
            phi = Parameter(f'φ_{i}')
            self.circuit_parameters.append(phi)
            circuit.rz(phi, q_reg[i + self.n_qubits])
        
        # Add measurements for all qubits
        circuit.measure(q_reg, c_reg)
        
        # Add barrier before measurements to prevent optimization across measurement
        circuit.barrier()
        
        return circuit

    def _execute_quantum_circuit(self, circuit, shots=1000):
        """Execute quantum circuit with parameter binding and error mitigation
        
        Args:
            circuit (QuantumCircuit): Circuit to execute
            shots (int): Number of shots for execution
            
        Returns:
            ndarray: Resulting quantum state vector
        """
        # Configure backend
        backend = AerSimulator()
        
        # Bind parameters if they exist
        if hasattr(self, 'circuit_parameters') and circuit.parameters:
            # Generate random parameter values (you might want to make this more sophisticated)
            param_values = np.random.uniform(-np.pi, np.pi, len(self.circuit_parameters))
            param_dict = dict(zip(self.circuit_parameters, param_values))
            circuit = circuit.assign_parameters(param_dict)
        
        # Add barrier before measurements if not already present
        if not (circuit.data and circuit.data[-1].operation.name == 'barrier'):
            circuit.barrier()
        
        # Multiple circuit executions for error mitigation
        results = []
        for _ in range(3):  # Multiple runs for better accuracy
            job = backend.run(circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()
            # Convert counts to vector
            vector = self._counts_to_vector(counts)
            results.append(vector)
        
        # Average results for error mitigation
        return np.mean(results, axis=0)

    def _counts_to_vector(self, counts):
        """Convert quantum measurement counts to vector"""
        vector = np.zeros(2**self.n_qubits)
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            # Ensure bitstring is the correct length
            bitstring = bitstring.zfill(2 * self.n_qubits)  # Double the qubits as we're measuring both registers
            # Take only the relevant part of the bitstring for the output register
            output_bitstring = bitstring[self.n_qubits:]
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