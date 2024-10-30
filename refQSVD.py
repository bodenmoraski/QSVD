
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator, UnitarySimulator  
from qiskit_algorithms.optimizers import SPSA, ADAM
import matplotlib.pyplot as plt
from scipy.linalg import svd as classical_svd
import time

class VQSVD:
    def __init__(self, matrix: np.ndarray, rank: int, num_qubits: int, depth: int, lr: float, itr: int):
        self.matrix = np.array(matrix)
        self.rank = rank
        self.num_qubits = num_qubits
        self.circuit_depth = depth
        self.lr = lr
        self.max_iters = itr

        # Check if matrix size matches number of qubits
        if self.matrix.shape != (2**num_qubits, 2**num_qubits):
            raise ValueError(f"Matrix shape {self.matrix.shape} does not match the number of qubits {num_qubits}")

        # Initialize quantum circuits for U and V
        self.circuit_U = self.create_parameterized_circuit(prefix='U')
        self.circuit_V = self.create_parameterized_circuit(prefix='V')

        # Initialize parameters
        self.params_U = np.random.uniform(-np.pi, np.pi, len(self.circuit_U.parameters))
        self.params_V = np.random.uniform(-np.pi, np.pi, len(self.circuit_V.parameters))

        # Set up simulator
        self.backend = UnitarySimulator()

        # Compute true singular values
        self.true_singular_values = classical_svd(self.matrix, compute_uv=False)[:self.rank]
        print(f"True singular values: {self.true_singular_values}")

        # Lists to store loss and singular values over iterations
        self.loss_history = []
        self.singular_values_history = []

    def create_parameterized_circuit(self, prefix: str) -> QuantumCircuit:
        circuit = QuantumCircuit(self.num_qubits)
        for d in range(self.circuit_depth):
            for q in range(self.num_qubits):
                theta = Parameter(f'{prefix}_theta_{d}_{q}')
                circuit.ry(theta, q)
            if d < self.circuit_depth - 1:  # No entanglement on last layer
                for q in range(self.num_qubits - 1):
                    circuit.cx(q, q + 1)
        return circuit

    def get_unitary(self, circuit: QuantumCircuit, params: np.ndarray) -> np.ndarray:
        param_dict = dict(zip(circuit.parameters, params))
        bound_circuit = circuit.assign_parameters(param_dict)
        job = self.backend.run(bound_circuit)
        result = job.result()
        unitary = result.get_unitary()
        return np.array(unitary)

    def loss_function(self, params_U: np.ndarray, params_V: np.ndarray) -> float:
        U = self.get_unitary(self.circuit_U, params_U)
        V = self.get_unitary(self.circuit_V, params_V)
        product = np.conj(U.T) @ self.matrix @ V
        estimated_singular_values = np.sort(np.abs(np.diag(product)))[::-1][:self.rank]
        
        # Mean Squared Error between estimated and true singular values
        mse = np.mean((estimated_singular_values - self.true_singular_values) ** 2)
        
        return mse

    def objective(self, params):
        params_U = params[:len(self.circuit_U.parameters)]
        params_V = params[len(self.circuit_U.parameters):]
        return self.loss_function(params_U, params_V)

    def clip_gradients(self, gradients, max_norm=1.0):
        total_norm = np.linalg.norm(gradients)
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            return gradients * clip_coef
        return gradients

    def optimize(self):
        initial_lr = self.lr
        optimizer = ADAM(maxiter=self.max_iters, lr=initial_lr)
        params = np.concatenate([self.params_U, self.params_V])
        current_lr = initial_lr

        def objective_with_callback(parameters):
            nonlocal current_lr
            loss = self.objective(parameters)
            self.loss_history.append(loss)
            
            # Adaptive learning rate
            if len(self.loss_history) > 1:
                if self.loss_history[-1] > self.loss_history[-2]:
                    current_lr *= 0.5
                elif len(self.loss_history) % 10 == 0:
                    current_lr = min(initial_lr, current_lr * 1.1)
            
            # Create a new optimizer with the updated learning rate
            optimizer = ADAM(maxiter=self.max_iters, lr=current_lr)
            # Compute gradients
            eps = 1e-8
            grads = []
            for i in range(len(parameters)):
                params_plus = parameters.copy()
                params_plus[i] += eps
                loss_plus = self.objective(params_plus)
                grad = (loss_plus - loss) / eps
                grads.append(grad)
            
            # Clip gradients
            clipped_grads = self.clip_gradients(np.array(grads))
            
            # Update parameters
            #parameters -= current_lr * clipped_grads # testong removing this
            
            if len(self.loss_history) % 10 == 0:
                print(f"Iteration {len(self.loss_history)}, Loss: {loss:.6f}, LR: {current_lr:.6f}")
                U = self.get_unitary(self.circuit_U, parameters[:len(self.circuit_U.parameters)])
                V = self.get_unitary(self.circuit_V, parameters[len(self.circuit_U.parameters):])
                product = np.conj(U.T) @ self.matrix @ V
                current_singular_values = np.sort(np.abs(np.diag(product)))[::-1][:self.rank]
                self.singular_values_history.append(current_singular_values)
                print(f"Estimated singular values: {current_singular_values}")
                print(f"True singular values: {self.true_singular_values}")
                print(f"Relative error: {np.mean(np.abs(current_singular_values - self.true_singular_values) / self.true_singular_values):.6f}")
            
            return loss

        result = optimizer.minimize(fun=objective_with_callback, x0=params)

        self.params_U = result.x[:len(self.circuit_U.parameters)]
        self.params_V = result.x[len(self.circuit_U.parameters):]
        final_loss = result.fun

        return self.params_U, self.params_V, final_loss

    def plot_results(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history)
        plt.title('Loss History')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')

        plt.subplot(1, 2, 2)
        singular_values_history = np.array(self.singular_values_history)
        for i in range(self.rank):
            plt.plot(singular_values_history[:, i], label=f'SV {i+1}')
        plt.plot(np.tile(self.true_singular_values, (len(self.singular_values_history), 1)), '--', color='black')
        plt.title('Singular Values Evolution')
        plt.xlabel('Iteration')
        plt.ylabel('Singular Value')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def compare_with_classical_svd(self):
        """
        Compares the VQSVD results with classical SVD.

        Returns:
            dict: A dictionary containing comparison metrics.
        """
        # Classical SVD
        start_time = time.time()
        U_classical, s_classical, Vt_classical = classical_svd(self.matrix, full_matrices=False)
        classical_time = time.time() - start_time

        # VQSVD results
        training_time, vqsvd_singular_values, final_loss = self.optimize()
        vqsvd_singular_values = np.array(vqsvd_singular_values)
        classical_singular_values = s_classical[:self.rank]

        # Calculate relative error
        relative_error = np.mean(np.abs(vqsvd_singular_values - classical_singular_values) / classical_singular_values)

        # Theoretical Quantum Advantage Factor (Placeholder)
        quantum_advantage = np.log2(self.matrix.shape[0])  # Example: logarithmic advantage

        # Print results
        print(f"VQSVD Training Time: {training_time:.4f} seconds")
        print(f"Classical SVD Time: {classical_time:.4f} seconds")
        print(f"Final VQSVD Loss: {final_loss:.6f}")
        print(f"VQSVD Singular Values: {vqsvd_singular_values}")
        print(f"Classical Singular Values: {classical_singular_values}")
        print(f"Relative Error in Top {self.rank} Singular Values: {relative_error:.4f}")
        print(f"Theoretical Quantum Advantage Factor: {quantum_advantage:.2f}x")

        # Plot comparison
        self.plot_comparison(vqsvd_singular_values, classical_singular_values)
        self.plot_loss_history()

        return {
            "VQSVD Time": training_time,
            "Classical SVD Time": classical_time,
            "Relative Error": relative_error,
            "Quantum Advantage Factor": quantum_advantage,
            "VQSVD Singular Values": vqsvd_singular_values,
            "Classical Singular Values": classical_singular_values,
            "Final Loss": final_loss
        }

    def plot_comparison(self, vqsvd_values: np.ndarray, classical_values: np.ndarray):
        """
        Plots a comparison of singular values obtained from VQSVD and classical SVD.

        Args:
            vqsvd_values (np.ndarray): Singular values from VQSVD.
            classical_values (np.ndarray): Singular values from classical SVD.
        """
        indices = np.arange(1, self.rank + 1)
        width = 0.35  # Width of the bars

        plt.figure(figsize=(10, 6))
        plt.bar(indices - width/2, vqsvd_values, width, label='VQSVD', color='blue')
        plt.bar(indices + width/2, classical_values, width, label='Classical SVD', color='red')

        plt.xlabel('Singular Value Index')
        plt.ylabel('Singular Value Magnitude')
        plt.title('VQSVD vs Classical SVD: Top Singular Values')
        plt.xticks(indices)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_loss_history(self):
        """
        Plots the loss history over iterations.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.loss_history)), self.loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss History')
        plt.yscale('log')  # Use log scale for y-axis
        plt.grid(True)
        plt.show()

    def analyze_circuit_expressiveness(self):
        num_params = len(self.circuit_U.parameters) + len(self.circuit_V.parameters)
        matrix_size = self.matrix.size
        print(f"Number of circuit parameters: {num_params}")
        print(f"Matrix size: {matrix_size}")
        print(f"Ratio of parameters to matrix size: {num_params / matrix_size:.4f}")
    def plot_optimization_landscape(self):
        num_points = 20  # Reduced for faster computation
        param_range = np.linspace(-np.pi, np.pi, num_points)
        loss_landscape = np.zeros((num_points, num_points))
        for i, p1 in enumerate(param_range):
            for j, p2 in enumerate(param_range):
                params = np.array([p1, p2] * (len(self.params_U) // 2 + len(self.params_V) // 2))
                loss_landscape[i, j] = self.objective(params)
        plt.figure(figsize=(10, 8))
        plt.imshow(loss_landscape, extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower', aspect='auto')
        plt.colorbar(label='Loss')
        plt.title('Optimization Landscape')
        plt.xlabel('Parameter 1')
        plt.ylabel('Parameter 2')
        plt.show()

    def check_unitarity(self, params_U, params_V):
        U = self.get_unitary(self.circuit_U, params_U)
        V = self.get_unitary(self.circuit_V, params_V)
        U_unitarity_error = np.linalg.norm(U @ np.conj(U.T) - np.eye(U.shape[0]))
        V_unitarity_error = np.linalg.norm(V @ np.conj(V.T) - np.eye(V.shape[0]))
        print(f"U unitarity error: {U_unitarity_error:.6f}")
        print(f"V unitarity error: {V_unitarity_error:.6f}")

    def initialize_parameters(self, strategy='random'):
        if strategy == 'random':
            return np.random.uniform(-np.pi, np.pi, len(self.circuit_U.parameters) + len(self.circuit_V.parameters))
        elif strategy == 'zero':
            return np.zeros(len(self.circuit_U.parameters) + len(self.circuit_V.parameters))
        elif strategy == 'small':
            return np.random.uniform(-0.1, 0.1, len(self.circuit_U.parameters) + len(self.circuit_V.parameters))

"""

# Hyperparameter settings
RANK = 8
ITR = 1000  # Increased from 500
# Reasoning: More iterations allow for better convergence, especially with adaptive learning rate
# Alternative: Could be set to 2000 for even more thorough optimization, at the cost of longer runtime

LR = 0.1  # Decreased from 0.05
# Reasoning: A smaller initial learning rate can lead to more stable optimization
# The adaptive learning rate mechanism will adjust this during training
# Alternative: Could start with 0.005 for even more stability, or 0.02 for faster initial progress

NUM_QUBITS = 4  # Unchanged, as this is determined by the input matrix size

CIRCUIT_DEPTH = 5  # Increased from 3
# Reasoning: Deeper circuits can express more complex transformations
# This is especially important given the difficulty in approximating the singular values
# Alternative: Could be set to 4 for a balance between expressivity and optimization difficulty


np.random.seed(42)
M = np.random.randint(1, 10, size=(2**NUM_QUBITS, 2**NUM_QUBITS)) + 1j * np.random.randint(1, 10, size=(2**NUM_QUBITS, 2**NUM_QUBITS))

net = VQSVD(matrix=M, rank=RANK, num_qubits=NUM_QUBITS, depth=CIRCUIT_DEPTH, lr=LR, itr=ITR)

print("Circuit Expressiveness Analysis:")
net.analyze_circuit_expressiveness()

print("\nInitial Unitarity Check:")
net.check_unitarity(net.params_U, net.params_V)

print("\nOptimization Landscape:")
net.plot_optimization_landscape()

print("\nStarting Optimization:")
final_params_U, final_params_V, final_loss = net.optimize()

print("\nFinal Unitarity Check:")
net.check_unitarity(final_params_U, final_params_V)

print("\nFinal Loss Components:")
final_loss = net.loss_function(final_params_U, final_params_V)
print(f"Final Loss: {final_loss:.6f}")

# If you want to print more detailed components of the loss:
U = net.get_unitary(net.circuit_U, final_params_U)
V = net.get_unitary(net.circuit_V, final_params_V)
product = np.conj(U.T) @ net.matrix @ V
estimated_singular_values = np.sort(np.abs(np.diag(product)))[::-1][:net.rank]
print(f"Estimated singular values: {estimated_singular_values}")
print(f"True singular values: {net.true_singular_values}")
print(f"Mean Squared Error: {np.mean((estimated_singular_values - net.true_singular_values) ** 2):.6f}")

print("\nLoss History:")
plt.plot(net.loss_history)
plt.title('Loss History')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
"""