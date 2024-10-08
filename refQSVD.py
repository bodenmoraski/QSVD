import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import ADAM
import matplotlib.pyplot as plt

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
    return lr * np.exp(-decay_rate * itr)

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
        self.initial_lr = lr  # Store initial learning rate for decay
        self.lr = lr
        self.itr = itr
        self.current_iteration = 0  # To track iterations for dynamic depth and learning rate
        
        self.base_depth = depth  # Initial depth
        self.cir_depth = depth   # Current depth
        
        self.cir_U = U_theta(num_qubits, self.cir_depth)
        self.cir_V = U_theta(num_qubits, self.cir_depth)
        
        self.M = Operator(matrix).data
        self.weight = weights
        
        self.params = list(self.cir_U.parameters) + list(self.cir_V.parameters)
        
    def loss_func(self, params_U, params_V):
        # Compute U and V matrices directly from parameters
        U = self.compute_unitary(self.cir_U, params_U)
        V = self.compute_unitary(self.cir_V, params_V)
        
        product = np.real(np.conj(U.T) @ self.M @ V)
        diag_elements = np.diag(product)
        
        # Compute the loss as the sum of squared differences
        loss = np.sum((diag_elements - self.weight[:self.rank]) ** 2)
        
        return loss, diag_elements
    
    def compute_unitary(self, circuit, params):
        # Create a copy of the circuit
        bound_circuit = circuit.copy()
        
        # Assign parameters
        param_dict = dict(zip(bound_circuit.parameters, params))
        bound_circuit = bound_circuit.assign_parameters(param_dict)
        
        # Use Aer's unitary simulator to compute the unitary
        backend = AerSimulator(method='unitary')
        job = backend.run(bound_circuit)
        result = job.result()
        
        # Get the unitary from the result
        unitary = result.get_unitary()
        
        return unitary
    
    def get_matrix_U(self):
        return self.compute_unitary(self.cir_U, [0] * len(self.cir_U.parameters))

    def get_matrix_V(self):
        return self.compute_unitary(self.cir_V, [0] * len(self.cir_V.parameters))
    
    def increase_circuit_depth(self):
        """
        Dynamically increases the circuit depth by adding an additional layer of gates.
        Ensures that the new parameters are appended to the parameter list with unique names.
        """
        self.cir_depth += 1  # Increment depth
        
        # Create new circuits with only the additional layer
        new_layer_U = U_theta(len(self.cir_U.qubits), 1)
        new_layer_V = U_theta(len(self.cir_V.qubits), 1)
        
        # Rename the parameters in the new layer to avoid conflicts
        param_map_U = {param: Parameter(f"{param.name}_{self.cir_depth}") for param in new_layer_U.parameters}
        param_map_V = {param: Parameter(f"{param.name}_{self.cir_depth}") for param in new_layer_V.parameters}
        
        new_layer_U = new_layer_U.assign_parameters(param_map_U)
        new_layer_V = new_layer_V.assign_parameters(param_map_V)
        
        # Append the new layer to the existing circuits
        self.cir_U = self.cir_U.compose(new_layer_U)
        self.cir_V = self.cir_V.compose(new_layer_V)
        
        # Update parameters list
        self.params = list(self.cir_U.parameters) + list(self.cir_V.parameters)
    
    def train(self):
        loss_list, singular_value_list = [], []
        
        stopped = False 
        final_params = None
        
        def objective(params):
            nonlocal final_params
            nonlocal stopped
            nonlocal self
            
            # Update learning rate based on current iteration
            self.current_iteration += 1
            self.lr = ExponentiallyDecayingLearningRate(self.initial_lr, self.current_iteration)
            
            # Dynamically increase circuit depth every 50 iterations
            if self.current_iteration % 50 == 0:
                self.increase_circuit_depth()
                print(f"Circuit depth increased to {self.cir_depth} at iteration {self.current_iteration}.")
                # Reinitialize params if the circuit depth has changed
                params = np.random.random(len(self.params)) * 2 * np.pi
            
            params_U = params[:len(self.cir_U.parameters)]
            params_V = params[len(self.cir_U.parameters):]
            
            loss, singular_values = self.loss_func(params_U, params_V)
            
            # Store the loss and singular values
            loss_list.append(loss.real)
            singular_value_list.append(singular_values)
            
            # Print progress every 10 iterations
            if len(loss_list) % 10 == 0:
                print(f'iter: {len(loss_list)}, loss: {loss.real:.4f}, lr: {self.lr:.6f}')
                
            if loss.real < 0.2:
                print('Loss less than 0.2')
                raise StopIteration
            
            return loss.real
        
        initial_params = np.random.random(len(self.params)) * 2 * np.pi
        
        try:
            for _ in range(self.itr):
                # Create a new optimizer instance with the current learning rate
                optimizer = ADAM(maxiter=1, lr=self.lr)
                result = optimizer.minimize(fun=objective, x0=initial_params)
                initial_params = result.x  # Update initial_params for the next iteration
                
                if result.fun < 0.2:
                    print('Loss less than 0.2')
                    break
            
            final_params = initial_params
        except StopIteration:
            print('Training stopped early due to loss threshold being met.')
            self.itr = self.current_iteration  # Update iteration count
            stopped = True
        
        # Update the circuits with the final optimized parameters
        if stopped and final_params is None:
            final_params = initial_params
            
        param_dict_U = dict(zip(self.cir_U.parameters, final_params[:len(self.cir_U.parameters)]))
        param_dict_V = dict(zip(self.cir_V.parameters, final_params[len(self.cir_U.parameters):]))
        self.cir_U = self.cir_U.assign_parameters(param_dict_U)
        self.cir_V = self.cir_V.assign_parameters(param_dict_V)
        
        return loss_list, singular_value_list

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
U_learned = net.get_matrix_U()
V_dagger_learned = net.get_matrix_V().conj().T

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