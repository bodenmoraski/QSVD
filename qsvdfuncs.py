import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator, UnitarySimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from qiskit_algorithms.optimizers import SPSA, ADAM
import matplotlib.pyplot as plt
from scipy.linalg import svd as classical_svd
import time


def create_parameterized_circuit(num_qubits: int, circuit_depth: int, prefix: str) -> QuantumCircuit:
    """
    Creates a parameterized quantum circuit with specified depth and qubits.
    Incorporates rotation gates, entangling gates, and SWAP gates for expressiveness.
    """
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
                circuit.p(phi, q)  # Phase gate
            
        if d % 2 == 0 and d < circuit_depth - 1:
            for q in range(0, num_qubits - 1, 2):
                circuit.swap(q, q + 1)
    
    return circuit


def get_unitary(circuit: QuantumCircuit, params: np.ndarray) -> np.ndarray:
    """
    Calculates the unitary matrix of the quantum circuit without noise.
    """
    param_dict = dict(zip(circuit.parameters, params))
    bound_circuit = circuit.assign_parameters(param_dict)
    
    # Use Operator to get the unitary matrix
    unitary = Operator(bound_circuit).data
    return np.array(unitary)


def apply_noise_to_unitary(unitary: np.ndarray, noise_model: NoiseModel) -> np.ndarray:
    """
    Applies noise to the unitary matrix.
    This is a simplified approach and may not fully capture all noise effects.
    """
    # This is a placeholder for noise application
    # In a real scenario, you'd need to implement a more sophisticated noise application
    # based on your specific noise model
    noise_factor = 0.99  # Adjust this based on your noise model
    return unitary * noise_factor + (1 - noise_factor) * np.eye(unitary.shape[0])


def loss_function(matrix: np.ndarray, circuit_U: QuantumCircuit, circuit_V: QuantumCircuit, 
                  params_U: np.ndarray, params_V: np.ndarray, rank: int, 
                  noise_model: NoiseModel = None) -> float:
    """
    Computes the relative error between the estimated singular values from the quantum circuits
    and the true singular values of the input matrix.
    """
    U = get_unitary(circuit_U, params_U)
    V = get_unitary(circuit_V, params_V)
    
    if noise_model:
        U = apply_noise_to_unitary(U, noise_model)
        V = apply_noise_to_unitary(V, noise_model)
    
    product = np.conj(U.T) @ matrix @ V
    
    # Perform SVD on the product
    _, singular_values_estimated, _ = classical_svd(product)
    estimated_singular_values = singular_values_estimated[:rank]
    
    # Compute true singular values
    true_singular_values = classical_svd(matrix, compute_uv=False)[:rank]
    
    # Compute relative error
    relative_error = np.mean(np.abs(estimated_singular_values - true_singular_values) / (true_singular_values + 1e-8))
    
    return relative_error


def objective(params: np.ndarray, matrix: np.ndarray, circuit_U: QuantumCircuit, 
              circuit_V: QuantumCircuit, rank: int, noise_model: NoiseModel = None) -> float:
    """
    Objective function for optimization: computes the loss given the parameters.
    """
    params_U = params[:len(circuit_U.parameters)]
    params_V = params[len(circuit_U.parameters):]
    return loss_function(matrix, circuit_U, circuit_V, params_U, params_V, rank, noise_model=noise_model)


def optimize_vqsvd(matrix: np.ndarray, rank: int, num_qubits: int, circuit_depth: int, 
                   lr: float, max_iters: int, noise_model: NoiseModel = None) -> tuple:
    """
    Optimizes the parameters of the quantum circuits U and V to minimize the QSVD loss.
    Utilizes the ADAM optimizer for parameter updates.
    """
    circuit_U = create_parameterized_circuit(num_qubits, circuit_depth, 'U')
    circuit_V = create_parameterized_circuit(num_qubits, circuit_depth, 'V')
    
    params_U = np.random.uniform(-np.pi, np.pi, len(circuit_U.parameters))
    params_V = np.random.uniform(-np.pi, np.pi, len(circuit_V.parameters))
    params = np.concatenate([params_U, params_V])
    
    optimizer = ADAM(maxiter=max_iters, lr=lr)
    
    loss_history = []
    singular_values_history = []
    
    def objective_with_callback(parameters):
        loss = objective(parameters, matrix, circuit_U, circuit_V, rank, noise_model=noise_model)
        loss_history.append(loss)
        if len(loss_history) % 10 == 0:
            print(f"Iteration {len(loss_history)}, Loss: {loss:.6f}")
            U = get_unitary(circuit_U, parameters[:len(circuit_U.parameters)])
            V = get_unitary(circuit_V, parameters[len(circuit_U.parameters):])
            product = np.conj(U.T) @ matrix @ V
            _, singular_values_estimated, _ = classical_svd(product)
            current_singular_values = singular_values_estimated[:rank]
            singular_values_history.append(current_singular_values)
            true_singular_values = classical_svd(matrix, compute_uv=False)[:rank]
            print(f"Estimated singular values: {current_singular_values}")
            print(f"True singular values: {true_singular_values}")
        return loss
    
    result = optimizer.minimize(fun=objective_with_callback, x0=params)
    
    final_params_U = result.x[:len(circuit_U.parameters)]
    final_params_V = result.x[len(circuit_U.parameters):]
    final_loss = result.fun
    
    return final_params_U, final_params_V, final_loss, loss_history, singular_values_history


def plot_results(loss_history: list, singular_values_history: list, true_singular_values: np.ndarray):
    """
    Plots the loss history and the evolution of singular values during optimization.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot Loss History
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Loss')
    plt.title('Loss History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    
    # Plot Singular Values Evolution
    plt.subplot(1, 2, 2)
    singular_values_history = np.array(singular_values_history)
    for i in range(singular_values_history.shape[1]):
        plt.plot(singular_values_history[:, i], label=f'SV {i+1}')
    plt.plot(np.tile(true_singular_values, (len(singular_values_history), 1)).T, '--', color='black', label='True SVs')
    plt.title('Singular Values Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Singular Value')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def compare_with_classical_svd(matrix: np.ndarray, vqsvd_singular_values: np.ndarray, 
                               training_time: float, rank: int) -> dict:
    """
    Compares the QSVD results with classical SVD in terms of time and accuracy.
    """
    start_time = time.time()
    _, s_classical, _ = classical_svd(matrix, full_matrices=False)
    classical_time = time.time() - start_time

    classical_singular_values = s_classical[:rank]
    relative_error = np.mean(np.abs(vqsvd_singular_values - classical_singular_values) / (classical_singular_values + 1e-8))
    quantum_advantage = np.log2(matrix.shape[0])

    return {
        "VQSVD Time": training_time,
        "Classical SVD Time": classical_time,
        "Relative Error": relative_error,
        "Quantum Advantage Factor": quantum_advantage,
        "VQSVD Singular Values": vqsvd_singular_values,
        "Classical Singular Values": classical_singular_values
    }


def plot_singular_value_error(singular_values_history, true_singular_values):
    """
    Plots the error in singular value approximation over iterations.
    """
    plt.figure(figsize=(10, 6))
    errors = [
        np.abs(sv - true_sv) for sv, true_sv in zip(singular_values_history, true_singular_values)
    ]
    errors = np.array(errors)
    for i in range(errors.shape[1]):
        plt.plot(errors[:, i], label=f'SV {i+1} Error')
    plt.xlabel('Iteration')
    plt.ylabel('Singular Value Error')
    plt.title('Error in Singular Value Approximation Over Iterations')
    plt.legend()
    plt.show()


def run_vqsvd(matrix: np.ndarray, rank: int, num_qubits: int, circuit_depth: int, 
              lr: float, max_iters: int, noise_model: NoiseModel = None) -> dict:
    """
    Executes the entire QSVD process: optimization, plotting, and comparison.
    """
    start_time = time.time()
    final_params_U, final_params_V, final_loss, loss_history, singular_values_history = optimize_vqsvd(
        matrix, rank, num_qubits, circuit_depth, lr, max_iters, noise_model=noise_model
    )
    training_time = time.time() - start_time

    true_singular_values = classical_svd(matrix, compute_uv=False)[:rank]
    vqsvd_singular_values = singular_values_history[-1]

    plot_results(loss_history, singular_values_history, true_singular_values)
    plot_singular_value_error(singular_values_history, true_singular_values)  # New plot

    comparison = compare_with_classical_svd(matrix, vqsvd_singular_values, training_time, rank)
    comparison["Final Loss"] = final_loss
    comparison["Loss History"] = loss_history
    comparison["Singular Values History"] = singular_values_history

    return comparison


# Additional utility functions

def analyze_circuit_expressiveness(circuit_U: QuantumCircuit, circuit_V: QuantumCircuit, matrix_size: int):
    """
    Analyzes the expressiveness of the quantum circuits by comparing the number
    of parameters to the size of the matrix being decomposed.
    """
    num_params = len(circuit_U.parameters) + len(circuit_V.parameters)
    print(f"Number of circuit parameters: {num_params}")
    print(f"Matrix size: {matrix_size}")
    print(f"Ratio of parameters to matrix size: {num_params / matrix_size:.4f}")


def plot_optimization_landscape(matrix: np.ndarray, circuit_U: QuantumCircuit, circuit_V: QuantumCircuit, rank: int):
    """
    Plots the loss landscape over a grid of parameter values to visualize optimization terrain.
    """
    num_points = 20
    param_range = np.linspace(-np.pi, np.pi, num_points)
    loss_landscape = np.zeros((num_points, num_points))
    
    # Limit to two parameters for visualization
    if len(circuit_U.parameters) + len(circuit_V.parameters) < 2:
        raise ValueError("Not enough parameters to plot a 2D landscape.")
    
    param1 = circuit_U.parameters[0]
    param2 = circuit_U.parameters[1]
    
    for i, p1 in enumerate(param_range):
        for j, p2 in enumerate(param_range):
            params = np.zeros(len(circuit_U.parameters) + len(circuit_V.parameters))
            params[0] = p1
            params[1] = p2
            loss_landscape[i, j] = objective(params, matrix, circuit_U, circuit_V, rank)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(loss_landscape, extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower', aspect='auto')
    plt.colorbar(label='Loss')
    plt.title('Optimization Landscape')
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.show()


def qsvd_decompose(operator: np.ndarray) -> tuple:
    """
    Placeholder for QSVD decomposition of a quantum operator.
    Returns identity matrices as placeholders.
    """
    Q = np.identity(operator.shape[0])
    Sigma = np.diag(np.ones(operator.shape[0]))
    V_dagger = Q.T
    return Q, Sigma, V_dagger


def apply_qsvd_and_calibrate(circuit: QuantumCircuit, operator: np.ndarray) -> QuantumCircuit:
    """
    Decomposes the circuit's unitary operator using QSVD and applies calibration adjustments.
    Currently serves as a placeholder for future enhancements.
    """
    Q, Sigma, V_dagger = qsvd_decompose(operator)
    # Future work: Adjust the circuit based on decomposed matrices
    return circuit


def encode_matrix_as_state(matrix: np.ndarray, num_qubits: int) -> QuantumCircuit:
    """
    Encodes a classical matrix into a quantum state using amplitude encoding.
    """
    # Normalize the matrix
    norm = np.linalg.norm(matrix)
    normalized_matrix = matrix.flatten() / norm
    circuit = QuantumCircuit(num_qubits)
    circuit.initialize(normalized_matrix, range(num_qubits))
    return circuit


def get_gradient(circuit: QuantumCircuit, params: np.ndarray, matrix: np.ndarray, rank: int, 
                noise_model: NoiseModel = None) -> np.ndarray:
    """
    Estimates the gradient of the loss function with respect to circuit parameters
    using the parameter shift rule.
    """
    gradients = np.zeros_like(params)
    shift = np.pi / 2  # Shift for parameter shift rule
    
    for i in range(len(params)):
        params_shift_up = np.copy(params)
        params_shift_down = np.copy(params)
        params_shift_up[i] += shift
        params_shift_down[i] -= shift
        
        loss_up = loss_function(matrix, circuit, circuit, params_shift_up[:len(circuit.parameters)], 
                                params_shift_up[len(circuit.parameters):], rank, noise_model=noise_model)
        loss_down = loss_function(matrix, circuit, circuit, params_shift_down[:len(circuit.parameters)], 
                                  params_shift_down[len(circuit.parameters):], rank, noise_model=noise_model)
        
        gradients[i] = (loss_up - loss_down) / 2.0
    
    return gradients



