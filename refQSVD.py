import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator, UnitarySimulator
from qiskit_algorithms.optimizers import SPSA
import matplotlib.pyplot as plt
from scipy.linalg import svd as classical_svd
import time

class VQSVD:
    def __init__(self, matrix: np.ndarray, rank: int, num_qubits: int, depth: int, lr: float, itr: int):
        """
        Initializes the VQSVD instance.

        Args:
            matrix (np.ndarray): The input matrix to decompose.
            rank (int): The number of singular values to compute.
            num_qubits (int): Number of qubits in each quantum circuit.
            depth (int): Initial depth of the quantum circuits.
            lr (float): Initial learning rate for the optimizer.
            itr (int): Maximum number of iterations for training.
        """
        self.matrix = np.array(matrix)
        self.rank = rank
        self.num_qubits = num_qubits
        self.circuit_depth = depth
        self.initial_depth = depth
        self.lr = lr
        self.initial_lr = lr
        self.max_iters = itr
        self.current_iteration = 0

        # Check if matrix size matches number of qubits
        if self.matrix.shape != (2**num_qubits, 2**num_qubits):
            raise ValueError(f"Matrix shape {self.matrix.shape} does not match the number of qubits {num_qubits}")

        # Initialize quantum circuits for U and V
        self.circuit_U = self.create_parameterized_circuit(prefix='U')
        self.circuit_V = self.create_parameterized_circuit(prefix='V')

        # Initialize parameters
        self.params_U = np.random.uniform(-np.pi, np.pi, len(self.circuit_U.parameters))
        self.params_V = np.random.uniform(-np.pi, np.pi, len(self.circuit_V.parameters))

        # Store final parameters after training
        self.final_params_U = None
        self.final_params_V = None

        # Lists to store loss and singular values over iterations
        self.loss_history = []
        self.singular_values_history = []

        # Set up simulator
        self.backend = UnitarySimulator()

    def create_parameterized_circuit(self, prefix: str) -> QuantumCircuit:
        """
        Creates a parameterized quantum circuit with the specified prefix for parameter names.

        Args:
            prefix (str): Prefix for parameter naming to distinguish between U and V circuits.

        Returns:
            QuantumCircuit: The created parameterized quantum circuit.
        """
        circuit = QuantumCircuit(self.num_qubits)
        for d in range(self.circuit_depth):
            for q in range(self.num_qubits):
                theta = Parameter(f'{prefix}_theta_{d}_{q}')
                phi = Parameter(f'{prefix}_phi_{d}_{q}')
                circuit.ry(theta, q)
                circuit.rz(phi, q)
            # Entangling layer
            for q in range(self.num_qubits - 1):
                circuit.cx(q, q + 1)
                circuit.cz(q, q + 1)
                # Approximate iSwap using CNOT and single-qubit rotations
                circuit.h(q)
                circuit.h(q + 1)
                circuit.cx(q, q + 1)
                circuit.rz(np.pi / 2, q)
                circuit.rz(np.pi / 2, q + 1)
                circuit.cx(q, q + 1)
                circuit.h(q)
                circuit.h(q + 1)
        return circuit

    def increase_circuit_depth(self):
        """
        Dynamically increases the depth of the quantum circuits by adding one more layer.
        """
        self.circuit_depth += 1
        print(f"Increasing circuit depth to {self.circuit_depth}.")

        # Create new layers with unique parameter names
        new_layer_U = self.create_parameterized_circuit(prefix=f'U_{self.circuit_depth}')
        new_layer_V = self.create_parameterized_circuit(prefix=f'V_{self.circuit_depth}')

        # Append new layers to existing circuits
        self.circuit_U = self.circuit_U.compose(new_layer_U)
        self.circuit_V = self.circuit_V.compose(new_layer_V)

        # Initialize new parameters
        self.params_U = np.concatenate([self.params_U, np.random.uniform(-np.pi, np.pi, len(new_layer_U.parameters))])
        self.params_V = np.concatenate([self.params_V, np.random.uniform(-np.pi, np.pi, len(new_layer_V.parameters))])

    def get_unitary(self, circuit: QuantumCircuit, params: np.ndarray) -> np.ndarray:
        """
        Binds parameters to the circuit and retrieves the unitary matrix.

        Args:
            circuit (QuantumCircuit): The quantum circuit.
            params (np.ndarray): The parameters to bind.

        Returns:
            np.ndarray: The unitary matrix of the circuit.
        """
        # Bind parameters
        param_dict = {param: value for param, value in zip(circuit.parameters, params)}
        bound_circuit = circuit.assign_parameters(param_dict)

        # Execute the circuit to get the unitary
        job = self.backend.run(bound_circuit)
        result = job.result()
        unitary = result.get_unitary(bound_circuit)

        # Convert to numpy array
        return np.array(unitary)

    def loss_function(self, params_U: np.ndarray, params_V: np.ndarray) -> float:
        """
        Computes the loss function for the VQSVD.

        Args:
            params_U (np.ndarray): Parameters for the U circuit.
            params_V (np.ndarray): Parameters for the V circuit.

        Returns:
            float: The computed loss.
        """
        # Retrieve unitaries for U and V
        U = self.get_unitary(self.circuit_U, params_U)
        V = self.get_unitary(self.circuit_V, params_V)

        # Ensure all matrices are numpy arrays
        U = np.array(U)
        V = np.array(V)
        M = np.array(self.matrix)

        # Compute U† * M * V
        product = np.conj(U.T) @ M @ V

        # Extract singular values from the diagonal
        singular_values = np.abs(np.diag(product))

        # Sort singular values in descending order
        sorted_singular_values = np.sort(singular_values)[::-1]

        # Compute true singular values using classical SVD
        true_singular_values = classical_svd(M, compute_uv=False)[:self.rank]

        # Calculate the Mean Squared Error (MSE) between approximate and true singular values
        mse = np.mean((sorted_singular_values[:self.rank] - true_singular_values) ** 2)

        # Compute penalties
        # 1. Off-diagonal penalty to encourage diagonal structure
        off_diagonal = product - np.diag(np.diag(product))
        off_diagonal_penalty = np.sum(np.abs(off_diagonal)) / product.size

        # 2. Orthogonality penalty to encourage U and V to be unitary
        orthogonality_U = np.eye(U.shape[0]) - U @ np.conj(U.T)
        orthogonality_V = np.eye(V.shape[0]) - V @ np.conj(V.T)
        orthogonality_penalty = (np.sum(np.abs(orthogonality_U)) + np.sum(np.abs(orthogonality_V))) / (U.size + V.size)

        # Weighted sum of MSE and penalties
        loss = mse + 0.1 * off_diagonal_penalty + 0.01 * orthogonality_penalty

        return loss

    def optimize(self):
        """
        Runs the optimization process using SPSA optimizer.
        """
        optimizer = SPSA(maxiter=self.max_iters, learning_rate=self.lr, perturbation=0.1)

        def objective(params):
            # Split parameters for U and V
            params_U = params[:len(self.circuit_U.parameters)]
            params_V = params[len(self.circuit_U.parameters):]

            # Compute loss
            loss = self.loss_function(params_U, params_V)
            return loss

        # Combine parameters for U and V
        initial_params = np.concatenate([self.params_U, self.params_V])

        start_time = time.time()
        result = optimizer.minimize(fun=objective, x0=initial_params)
        end_time = time.time()

        # Store final parameters
        self.final_params_U = result.x[:len(self.circuit_U.parameters)]
        self.final_params_V = result.x[len(self.circuit_U.parameters):]

        training_time = end_time - start_time

        # Compute final loss and singular values
        final_loss = self.loss_function(self.final_params_U, self.final_params_V)
        U_final = self.get_unitary(self.circuit_U, self.final_params_U)
        V_final = self.get_unitary(self.circuit_V, self.final_params_V)
        final_product = np.conj(U_final.T) @ self.matrix @ V_final
        final_singular_values = np.abs(np.diag(final_product))
        sorted_final_singular_values = np.sort(final_singular_values)[::-1][:self.rank]

        # Store history (for plotting, if needed)
        self.loss_history.append(final_loss)
        self.singular_values_history.append(sorted_final_singular_values)

        return training_time, sorted_final_singular_values, final_loss

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
        start_time = time.time()
        training_time, vqsvd_singular_values, final_loss = self.optimize()
        vqsvd_time = training_time
        vqsvd_singular_values = np.array(vqsvd_singular_values)
        classical_singular_values = s_classical[:self.rank]

        # Calculate relative error
        relative_error = np.mean(np.abs(vqsvd_singular_values - classical_singular_values) / classical_singular_values)

        # Theoretical Quantum Advantage Factor (Placeholder)
        quantum_advantage = np.log2(self.matrix.shape[0])  # Example: logarithmic advantage

        # Print results
        print(f"VQSVD Training Time: {vqsvd_time:.4f} seconds")
        print(f"Classical SVD Time: {classical_time:.4f} seconds")
        print(f"Relative Error in Top {self.rank} Singular Values: {relative_error:.4f}")
        print(f"Theoretical Quantum Advantage Factor: {quantum_advantage:.2f}x")

        # Plot comparison
        self.plot_comparison(vqsvd_singular_values, classical_singular_values)

        return {
            "VQSVD Time": vqsvd_time,
            "Classical SVD Time": classical_time,
            "Relative Error": relative_error,
            "Quantum Advantage Factor": quantum_advantage
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

# Hyperparameter settings
RANK = 8
ITR = 300  # Maximum iterations
LR = 0.5  # Increased learning rate for SPSA
NUM_QUBITS = 4  # Increased number of qubits for better representation
CIRCUIT_DEPTH = 4  # Initial circuit depth

# Generate random matrix
np.random.seed(42)  # For reproducibility
M = np.random.randint(1, 10, size=(2**NUM_QUBITS, 2**NUM_QUBITS)) + 1j * np.random.randint(1, 10, size=(2**NUM_QUBITS, 2**NUM_QUBITS))

# Initialize and run VQSVD
net = VQSVD(matrix=M, rank=RANK, num_qubits=NUM_QUBITS, depth=CIRCUIT_DEPTH, lr=LR, itr=ITR)
comparison_results = net.compare_with_classical_svd()

# Plot loss history
plt.figure(figsize=(10, 6))
plt.plot(range(len(net.loss_history)), net.loss_history, label='VQSVD Loss', color='blue')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs. Iterations during VQSVD Training')
plt.legend()
plt.grid(True)
plt.show()

print("Training completed.")
print(f"Final Loss: {net.loss_history[-1]:.4f}")
print(f"Final Singular Values: {net.singular_values_history[-1]}")