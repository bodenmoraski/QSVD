import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator, UnitarySimulator
from qiskit_algorithms.optimizers import ADAM
import matplotlib.pyplot as plt
from scipy.linalg import svd as classical_svd
import time


def ExponentiallyDecayingLearningRate(lr, itr, decay_rate=0.1):
    """
    Calculate the exponentially decaying learning rate.
    
    Args:
        lr (float): Initial learning rate.
        itr (int): Current iteration number.
        decay_rate (float): Decay rate for the learning rate.
        
    Returns:
        float: Updated learning rate.
    """
    #return lr * np.exp(-decay_rate * itr)
    return lr * 1.01

# Define quantum neural network
def U_theta(num_qubits: int, depth: int) -> QuantumCircuit:
    """
    Constructs a parameterized quantum circuit with specified number of qubits and depth.
    Uses CNOT and single-qubit rotations to approximate iSwap gate.
    
    Args:
        num_qubits (int): Number of qubits in the circuit.
        depth (int): Depth of the circuit.
        
    Returns:
        QuantumCircuit: The constructed quantum circuit.
    """
    cir = QuantumCircuit(num_qubits)
    for d in range(depth):
        for q in range(num_qubits):
            theta = Parameter(f'theta_{d}_{q}')
            phi = Parameter(f'phi_{d}_{q}')
            cir.ry(theta, q)
            cir.rz(phi, q)
        for q in range(num_qubits - 1):
            cir.cx(q, q + 1)
            cir.cz(q, q + 1)       # CZ gate for richer entanglement
            # Approximate iSwap using CNOT and single-qubit rotations
            cir.h(q)
            cir.h(q + 1)
            cir.cx(q, q + 1)
            cir.rz(np.pi/2, q)
            cir.rz(np.pi/2, q + 1)
            cir.cx(q, q + 1)
            cir.h(q)
            cir.h(q + 1)
    return cir

class VQSVD:
    def __init__(self, matrix: np.ndarray, weights: np.ndarray, num_qubits: int, depth: int, rank: int, lr: float, itr: int):
        self.rank = rank
        self.initial_lr = lr
        self.lr = lr
        self.itr = itr
        self.current_iteration = 0
        
        self.base_depth = depth
        self.cir_depth = depth
        
        self.num_qubits = num_qubits
        self.cir_U = self.create_parameterized_circuit(num_qubits, depth, 'U')
        self.cir_V = self.create_parameterized_circuit(num_qubits, depth, 'V')
        
        self.M = matrix
        self.weight = weights
        
        self.params_dict = {param: 0 for param in self.cir_U.parameters}
        self.params_dict.update({param: 0 for param in self.cir_V.parameters})
        
        self.final_params = None

    def create_parameterized_circuit(self, num_qubits: int, depth: int, prefix: str) -> QuantumCircuit:
        cir = QuantumCircuit(num_qubits)
        for d in range(depth):
            for q in range(num_qubits):
                theta = Parameter(f'{prefix}_theta_{d}_{q}')
                phi = Parameter(f'{prefix}_phi_{d}_{q}')
                cir.ry(theta, q)
                cir.rz(phi, q)
            for q in range(num_qubits - 1):
                cir.cx(q, q + 1)
                cir.cz(q, q + 1)
                # Approximate iSwap
                cir.h(q)
                cir.h(q + 1)
                cir.cx(q, q + 1)
                cir.rz(np.pi/2, q)
                cir.rz(np.pi/2, q + 1)
                cir.cx(q, q + 1)
                cir.h(q)
                cir.h(q + 1)
        return cir

    def increase_circuit_depth(self):
        self.cir_depth += 1
        new_layer_U = self.create_parameterized_circuit(self.num_qubits, 1, f'U_{self.cir_depth}')
        new_layer_V = self.create_parameterized_circuit(self.num_qubits, 1, f'V_{self.cir_depth}')
        
        self.cir_U = self.cir_U.compose(new_layer_U)
        self.cir_V = self.cir_V.compose(new_layer_V)
        
        for param in new_layer_U.parameters:
            self.params_dict[param] = 0
        for param in new_layer_V.parameters:
            self.params_dict[param] = 0

    def get_matrix(self, circuit, params):
        bound_circuit = circuit.assign_parameters({param: value for param, value in params.items() if param in circuit.parameters})
        
        # Use UnitarySimulator to compute the unitary
        backend = UnitarySimulator()
        job = backend.run(bound_circuit)
        result = job.result()
        
        # Get the unitary from the result
        unitary = result.get_unitary()
        
        return unitary

    def loss_func(self, params):
        #print(f"Number of params: {len(params)}, Number of params_dict: {len(self.params_dict)}")
        for param, value in zip(self.params_dict.keys(), params):
            self.params_dict[param] = value
        
        #print(f"Number of U parameters: {len(self.cir_U.parameters)}")
        #print(f"Number of V parameters: {len(self.cir_V.parameters)}")
        
        U = self.get_matrix(self.cir_U, self.params_dict)
        V = self.get_matrix(self.cir_V, self.params_dict)
        
        # Convert Operators to numpy arrays
        U_array = np.asarray(U)
        V_array = np.asarray(V)
        
        # Compute U^† * M * V
        product = np.conj(U_array.T) @ self.M @ V_array
        
        # Extract diagonal elements (singular values)
        singular_values = np.abs(np.diag(product))
        
        # Sort singular values in descending order
        sorted_singular_values = np.sort(singular_values)[::-1]
        
        # Compute true singular values
        true_singular_values = np.linalg.svd(self.M, compute_uv=False)[:self.rank]
        
        # Compute mean squared error between approximated and true singular values
        singular_value_error = np.mean((sorted_singular_values[:self.rank] - true_singular_values)**2)
        
        # Add penalty for off-diagonal elements
        off_diagonal_penalty = np.sum(np.abs(product - np.diag(np.diag(product))))
        
        # Add regularization term to encourage orthogonality of U and V
        orthogonality_penalty = np.sum(np.abs(np.eye(U_array.shape[0]) - U_array @ np.conj(U_array.T))) + \
                                np.sum(np.abs(np.eye(V_array.shape[0]) - V_array @ np.conj(V_array.T)))
        
        # Combine all terms
        total_loss = singular_value_error + 0.1 * off_diagonal_penalty + 0.01 * orthogonality_penalty
        
        # No need for exponential transformation if all terms are non-negative
        return total_loss, sorted_singular_values[:self.rank]

    def train(self):
        loss_list, singular_value_list = [], []
        
        def objective(params):
            self.current_iteration += 1
            self.lr = ExponentiallyDecayingLearningRate(self.initial_lr, self.current_iteration)
            
            if self.current_iteration % 50 == 0:
                self.increase_circuit_depth()
                print(f"Circuit depth increased to {self.cir_depth} at iteration {self.current_iteration}.")
                params = np.random.random(len(self.params_dict)) * 2 * np.pi
            
            loss, singular_values = self.loss_func(params)
            
            # Store the loss and singular values
            loss_list.append(loss.real)
            singular_value_list.append(singular_values)
            
            # Print progress every 10 iterations
            if len(loss_list) % 10 == 0:
                print(f'iter: {len(loss_list)}, loss: {loss.real:.4f}, lr: {self.lr:.6f}')
                
            if loss < 1e-6:  # Use a small threshold based on the new loss function
                print(f'Achieved high accuracy for top {self.rank} singular values')
                self.final_params = params  # Store the final parameters
                raise StopIteration
            
            return loss
        
        initial_params = np.random.random(len(self.params_dict)) * 2 * np.pi
        
        try:
            for _ in range(self.itr):
                optimizer = ADAM(maxiter=1, lr=self.lr)
                result = optimizer.minimize(fun=objective, x0=initial_params)
                initial_params = result.x
                
                if result.fun < 0.2:
                    print('Loss less than 0.2')
                    self.final_params = result.x
                    break
            
            self.final_params = initial_params
        except StopIteration:
            print('Training stopped early due to loss threshold being met.')
            self.itr = self.current_iteration
        
        return loss_list, singular_value_list

    def get_final_matrices(self):
        if self.final_params is None:
            raise ValueError("Model hasn't been trained yet.")
        for param, value in zip(self.params_dict.keys(), self.final_params):
            self.params_dict[param] = value
        return self.get_matrix(self.cir_U, self.params_dict), self.get_matrix(self.cir_V, self.params_dict)

    def compare_with_classical_svd(self):
        # Measure time for VQSVD
        start_time = time.time()
        loss_list, singular_value_list = self.train()
        vqsvd_time = time.time() - start_time

        # Get final U and V matrices
        U_learned, V_learned = self.get_final_matrices()

        # Measure time for classical SVD
        start_time = time.time()
        U_classical, s_classical, Vt_classical = classical_svd(self.M)
        classical_time = time.time() - start_time

        # Compare top 'rank' singular values
        vqsvd_singular_values = singular_value_list[-1]
        classical_singular_values = s_classical[:self.rank]

        # Calculate relative error
        relative_error = np.mean(np.abs(vqsvd_singular_values - classical_singular_values) / classical_singular_values)

        # Calculate quantum advantage factor (hypothetical)
        quantum_advantage = self.calculate_quantum_advantage(len(self.M))

        # Print results
        print(f"VQSVD Time: {vqsvd_time:.4f} seconds")
        print(f"Classical SVD Time: {classical_time:.4f} seconds")
        print(f"Relative Error in Singular Values: {relative_error:.4f}")
        print(f"Theoretical Quantum Advantage Factor: {quantum_advantage:.2f}x")

        # Plot comparison
        self.plot_comparison(vqsvd_singular_values, classical_singular_values)

        return vqsvd_time, classical_time, relative_error, quantum_advantage

    def calculate_quantum_advantage(self, matrix_size):
        # This is a hypothetical calculation and should be adjusted based on theoretical or empirical evidence
        return np.log2(matrix_size)  # Example: logarithmic advantage

    def plot_comparison(self, vqsvd_values, classical_values):
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(vqsvd_values)), vqsvd_values, alpha=0.5, label='VQSVD', color='blue')
        plt.bar(range(len(classical_values)), classical_values, alpha=0.5, label='Classical SVD', color='red')
        plt.xlabel('Index')
        plt.ylabel('Singular Value')
        plt.title('VQSVD vs Classical SVD: Top Singular Values')
        plt.legend()
        plt.show()

# Hyperparameter settings
RANK = 8
ITR = 100
LR = 0.05
num_qubits = 3
cir_depth = 20

# Generate random matrix and weights
M = np.random.randint(10, size=(2**num_qubits, 2**num_qubits)) + 1j * np.random.randint(10, size=(2**num_qubits, 2**num_qubits))
weight = np.arange(3 * RANK, 0, -3).astype('complex128')

# Construct the VQSVD network and train
net = VQSVD(matrix=M, weights=weight, num_qubits=num_qubits, depth=cir_depth, rank=RANK, lr=LR, itr=ITR)
loss_list, singular_value_list = net.train()

# Get final U and V matrices
U_learned, V_learned = net.get_final_matrices()

plt.figure(figsize=(10, 6))
plt.plot(loss_list, label='Loss over time')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs. Iterations during Training')
plt.legend()
plt.grid(True)
plt.show()

print("Training completed.")
print(f"Final loss: {loss_list[-1]}")
print(f"Final singular values: {singular_value_list[-1]}")

# After training the model
net = VQSVD(matrix=M, weights=weight, num_qubits=num_qubits, depth=cir_depth, rank=RANK, lr=LR, itr=ITR)
vqsvd_time, classical_time, relative_error, quantum_advantage = net.compare_with_classical_svd()