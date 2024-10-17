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

def create_parameterized_circuit(num_qubits: int, circuit_depth: int, prefix: str) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    for d in range(circuit_depth):
        for q in range(num_qubits):
            theta_x = Parameter(f'{prefix}_theta_x_{d}_{q}')
            theta_y = Parameter(f'{prefix}_theta_y_{d}_{q}')
            theta_z = Parameter(f'{prefix}_theta_z_{d}_{q}')
            circuit.rx(theta_x, q)
            circuit.ry(theta_y, q)
            circuit.rz(theta_z, q)
        
        if d < circuit_depth - 1:  # No entanglement on last layer
            for q in range(num_qubits - 1):
                circuit.cx(q, q + 1)
            for q in range(num_qubits - 2, -1, -1):
                circuit.cx(q, q + 1)
            
            for q in range(num_qubits):
                phi = Parameter(f'{prefix}_phi_{d}_{q}')
                circuit.p(phi, q)  # Changed from u1 to p
        
        if d % 2 == 0 and d < circuit_depth - 1:
            for q in range(0, num_qubits - 1, 2):
                circuit.swap(q, q + 1)
    
    return circuit

def get_unitary(circuit: QuantumCircuit, params: np.ndarray) -> np.ndarray:
    backend = UnitarySimulator()
    param_dict = dict(zip(circuit.parameters, params))
    bound_circuit = circuit.assign_parameters(param_dict)
    job = backend.run(bound_circuit)
    result = job.result()
    unitary = result.get_unitary()
    return np.array(unitary)

def loss_function(matrix: np.ndarray, circuit_U: QuantumCircuit, circuit_V: QuantumCircuit, 
                  params_U: np.ndarray, params_V: np.ndarray, rank: int) -> float:
    U = get_unitary(circuit_U, params_U)
    V = get_unitary(circuit_V, params_V)
    product = np.conj(U.T) @ matrix @ V
    estimated_singular_values = np.sort(np.abs(np.diag(product)))[::-1][:rank]
    true_singular_values = classical_svd(matrix, compute_uv=False)[:rank]
    mse = np.mean((estimated_singular_values - true_singular_values) ** 2)
    return mse

def objective(params: np.ndarray, matrix: np.ndarray, circuit_U: QuantumCircuit, 
              circuit_V: QuantumCircuit, rank: int) -> float:
    params_U = params[:len(circuit_U.parameters)]
    params_V = params[len(circuit_U.parameters):]
    return loss_function(matrix, circuit_U, circuit_V, params_U, params_V, rank)

def optimize_vqsvd(matrix: np.ndarray, rank: int, num_qubits: int, circuit_depth: int, 
                   lr: float, max_iters: int) -> tuple:
    circuit_U = create_parameterized_circuit(num_qubits, circuit_depth, 'U')
    circuit_V = create_parameterized_circuit(num_qubits, circuit_depth, 'V')
    
    params_U = np.random.uniform(-np.pi, np.pi, len(circuit_U.parameters))
    params_V = np.random.uniform(-np.pi, np.pi, len(circuit_V.parameters))
    params = np.concatenate([params_U, params_V])
    
    optimizer = ADAM(maxiter=max_iters, lr=lr)
    
    loss_history = []
    singular_values_history = []
    
    def objective_with_callback(parameters):
        loss = objective(parameters, matrix, circuit_U, circuit_V, rank)
        loss_history.append(loss)
        if len(loss_history) % 10 == 0:
            print(f"Iteration {len(loss_history)}, Loss: {loss:.6f}")
            U = get_unitary(circuit_U, parameters[:len(circuit_U.parameters)])
            V = get_unitary(circuit_V, parameters[len(circuit_U.parameters):])
            product = np.conj(U.T) @ matrix @ V
            current_singular_values = np.sort(np.abs(np.diag(product)))[::-1][:rank]
            singular_values_history.append(current_singular_values)
            print(f"Estimated singular values: {current_singular_values}")
        return loss
    
    result = optimizer.minimize(fun=objective_with_callback, x0=params)
    
    final_params_U = result.x[:len(circuit_U.parameters)]
    final_params_V = result.x[len(circuit_U.parameters):]
    final_loss = result.fun
    
    return final_params_U, final_params_V, final_loss, loss_history, singular_values_history

def plot_results(loss_history: list, singular_values_history: list, true_singular_values: np.ndarray):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('Loss History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')

    plt.subplot(1, 2, 2)
    singular_values_history = np.array(singular_values_history)
    for i in range(singular_values_history.shape[1]):
        plt.plot(singular_values_history[:, i], label=f'SV {i+1}')
    plt.plot(np.tile(true_singular_values, (len(singular_values_history), 1)), '--', color='black')
    plt.title('Singular Values Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Singular Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

def compare_with_classical_svd(matrix: np.ndarray, vqsvd_singular_values: np.ndarray, 
                               training_time: float, rank: int) -> dict:
    start_time = time.time()
    _, s_classical, _ = classical_svd(matrix, full_matrices=False)
    classical_time = time.time() - start_time

    classical_singular_values = s_classical[:rank]
    relative_error = np.mean(np.abs(vqsvd_singular_values - classical_singular_values) / classical_singular_values)
    quantum_advantage = np.log2(matrix.shape[0])

    return {
        "VQSVD Time": training_time,
        "Classical SVD Time": classical_time,
        "Relative Error": relative_error,
        "Quantum Advantage Factor": quantum_advantage,
        "VQSVD Singular Values": vqsvd_singular_values,
        "Classical Singular Values": classical_singular_values
    }

def run_vqsvd(matrix: np.ndarray, rank: int, num_qubits: int, circuit_depth: int, 
              lr: float, max_iters: int) -> dict:
    start_time = time.time()
    final_params_U, final_params_V, final_loss, loss_history, singular_values_history = optimize_vqsvd(
        matrix, rank, num_qubits, circuit_depth, lr, max_iters
    )
    training_time = time.time() - start_time

    true_singular_values = classical_svd(matrix, compute_uv=False)[:rank]
    vqsvd_singular_values = singular_values_history[-1]

    plot_results(loss_history, singular_values_history, true_singular_values)

    comparison = compare_with_classical_svd(matrix, vqsvd_singular_values, training_time, rank)
    comparison["Final Loss"] = final_loss
    comparison["Loss History"] = loss_history
    comparison["Singular Values History"] = singular_values_history

    return comparison

# Additional utility functions

def analyze_circuit_expressiveness(circuit_U: QuantumCircuit, circuit_V: QuantumCircuit, matrix_size: int):
    num_params = len(circuit_U.parameters) + len(circuit_V.parameters)
    print(f"Number of circuit parameters: {num_params}")
    print(f"Matrix size: {matrix_size}")
    print(f"Ratio of parameters to matrix size: {num_params / matrix_size:.4f}")

def plot_optimization_landscape(matrix: np.ndarray, circuit_U: QuantumCircuit, circuit_V: QuantumCircuit, rank: int):
    num_points = 20
    param_range = np.linspace(-np.pi, np.pi, num_points)
    loss_landscape = np.zeros((num_points, num_points))
    for i, p1 in enumerate(param_range):
        for j, p2 in enumerate(param_range):
            params = np.array([p1, p2] * (len(circuit_U.parameters) // 2 + len(circuit_V.parameters) // 2))
            loss_landscape[i, j] = objective(params, matrix, circuit_U, circuit_V, rank)
    plt.figure(figsize=(10, 8))
    plt.imshow(loss_landscape, extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower', aspect='auto')
    plt.colorbar(label='Loss')
    plt.title('Optimization Landscape')
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.show()

def check_unitarity(circuit_U: QuantumCircuit, circuit_V: QuantumCircuit, params_U: np.ndarray, params_V: np.ndarray):
    U = get_unitary(circuit_U, params_U)
    V = get_unitary(circuit_V, params_V)
    U_unitarity_error = np.linalg.norm(U @ np.conj(U.T) - np.eye(U.shape[0]))
    V_unitarity_error = np.linalg.norm(V @ np.conj(V.T) - np.eye(V.shape[0]))
    print(f"U unitarity error: {U_unitarity_error:.6f}")
    print(f"V unitarity error: {V_unitarity_error:.6f}")
