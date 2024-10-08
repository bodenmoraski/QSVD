import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import ADAM
import matplotlib.pyplot as plt

def ExponentiallyDecayingLearningRate(lr, itr):
    return lr * np.exp(-0.1 * itr)

# Define quantum neural network
def U_theta(num_qubits: int, depth: int) -> QuantumCircuit:
    cir = QuantumCircuit(num_qubits)
    for d in range(depth):
        for q in range(num_qubits):
            theta = Parameter(f'theta_{d}_{q}')
            phi = Parameter(f'phi_{d}_{q}')
            cir.ry(theta, q)
            cir.rz(phi, q)
        for q in range(num_qubits - 1):
            cir.cx(q, q + 1)
            cir.cz(q, q + 1)
    return cir

class VQSVD:
    def __init__(self, matrix: np.ndarray, weights: np.ndarray, num_qubits: int, depth: int, rank: int, lr: float, itr: int):
        self.rank = rank
        self.lr = lr
        self.itr = itr
        
        self.cir_U = U_theta(num_qubits, depth)
        self.cir_V = U_theta(num_qubits, depth)
        
        self.M = Operator(matrix).data
        self.weight = weights
        
        self.params = list(self.cir_U.parameters) + list(self.cir_V.parameters)
        
    def loss_func(self, cir_U, cir_V):
        # Compute the matrix products
        U = Operator(cir_U).data
        V = Operator(cir_V).data
        
        product = np.real(np.conj(U.T) @ self.M @ V)
        diag_elements = np.diag(product)
        
        # Compute the loss as the sum of squared differences
        loss = np.sum((diag_elements - self.weight[:self.rank]) ** 2)
        
        return loss, diag_elements
    
    def get_matrix_U(self):
        return Operator(self.cir_U).data
    
    def get_matrix_V(self):
        return Operator(self.cir_V).data
    
    def train(self):
        loss_list, singular_value_list = [], []
        optimizer = ADAM(maxiter=self.itr, lr=self.lr)
        
        stopped = False 
        final_params = None
        
        def objective(params):
            param_dict_U = dict(zip(self.cir_U.parameters, params[:len(self.cir_U.parameters)]))
            param_dict_V = dict(zip(self.cir_V.parameters, params[len(self.cir_U.parameters):]))
            
            cir_U = self.cir_U.assign_parameters(param_dict_U)
            cir_V = self.cir_V.assign_parameters(param_dict_V)
            
            loss, singular_values = self.loss_func(cir_U, cir_V)
            
            # Store the loss and singular values
            loss_list.append(loss.real)
            singular_value_list.append(singular_values)
            
            # Print progress every 10 iterations
            if len(loss_list) % 10 == 0:
                print(f'iter: {len(loss_list) / 10}, loss: {loss.real:.4f}')
                
            if loss.real < 0.2:
                print('loss less than 0.2')
                raise StopIteration
            
            # Print loss at every iteration
            print(f'iter: {len(loss_list)}, loss: {loss.real:.4f}')
            
            return loss.real
        
        initial_params = np.random.random(len(self.params)) * 2 * np.pi
        
        try:
            result = optimizer.minimize(fun=objective, x0=initial_params)
            final_params = result.x
        except StopIteration:
            print('Training stopped early due to loss threshold being met.')
            self.itr = 0
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

