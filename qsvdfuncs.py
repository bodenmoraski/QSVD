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
from qiskit.quantum_info import Operator
from qiskit.exceptions import QiskitError


def create_parameterized_circuit(num_qubits: int, circuit_depth: int, prefix: str) -> QuantumCircuit:
    """
    Creates a parameterized quantum circuit with specified depth and qubits.
    Incorporates rotation gates, entangling gates, and SWAP gates for expressiveness.
    """
    circuit = QuantumCircuit(num_qubits)
    
    # Reduce parameters dramatically
    for layer in range(min(circuit_depth, 2)):
        # Single rotation per qubit per layer
        for q in range(num_qubits):
            theta = Parameter(f'{prefix}_theta_{layer}_{q}')
            circuit.ry(theta, q)
        
        # Entangling layer
        if layer < circuit_depth - 1:
            for q in range(num_qubits - 1):
                circuit.cx(q, q + 1)
    
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
    Computes quantum-native loss based on three principles:
    1. Maximizing singular values (energy maximization)
    2. Maintaining orthogonality
    3. Preserving matrix properties (trace norm, Frobenius norm)
    """
    U = get_unitary(circuit_U, params_U)
    V = get_unitary(circuit_V, params_V)
    
    if noise_model:
        U = apply_noise_to_unitary(U, noise_model)
        V = apply_noise_to_unitary(V, noise_model)
    
    loss = 0.0
    M = matrix
    
    # 1. Energy Maximization Term
    energy_terms = []
    for i in range(rank):
        u_i = U[:, i]
        v_i = V[:, i]
        energy = np.abs(u_i.conj() @ M @ v_i)
        energy_terms.append(energy)
    
    # Sort energy terms to encourage proper ordering of singular values
    energy_terms.sort(reverse=True)
    energy_loss = -sum(energy_terms[i] * (rank - i) for i in range(rank))
    
    # 2. Orthogonality Terms
    ortho_loss_U = 0.0
    ortho_loss_V = 0.0
    for i in range(rank):
        for j in range(i):
            ortho_loss_U += np.abs(U[:, i].conj() @ U[:, j])**2
            ortho_loss_V += np.abs(V[:, i].conj() @ V[:, j])**2
    
    # 3. Matrix Property Preservation
    # Compute difference between input and reconstructed matrix properties
    reconstructed = U @ np.diag(energy_terms) @ V.conj().T
    trace_diff = np.abs(np.trace(M) - np.trace(reconstructed))
    frob_diff = np.abs(np.linalg.norm(M, 'fro') - np.linalg.norm(reconstructed, 'fro'))
    
    # Weighted combination of all terms
    alpha = 1.0  # Energy weight
    beta = 2.0   # Orthogonality weight
    gamma = 0.5  # Property preservation weight
    
    total_loss = (alpha * energy_loss + 
                  beta * (ortho_loss_U + ortho_loss_V) +
                  gamma * (trace_diff + frob_diff))
    
    return total_loss


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
    Actually perform quantum SVD using variational circuits
    """
    # Initialize quantum circuits for U and V
    circuit_U = create_parameterized_circuit(num_qubits, circuit_depth, 'U')
    circuit_V = create_parameterized_circuit(num_qubits, circuit_depth, 'V')
    
    # Initialize parameters randomly
    params = initialize_random_parameters(circuit_U, circuit_V)
    
    # Optimize using quantum measurements and classical feedback
    for iter in range(max_iters):
        # Measure quantum expectation values
        expectation = measure_quantum_expectation(circuit_U, circuit_V, params, matrix)
        
        # Update parameters using gradient descent
        params = update_parameters(params, expectation, lr)
        
        # Calculate current singular values from quantum state
        singular_values = extract_singular_values(circuit_U, circuit_V, params)
    
    return final_params, singular_values


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

    Args:
        matrix (np.ndarray): The matrix to decompose.
        rank (int): Number of singular values/components.
        num_qubits (int): Number of qubits.
        circuit_depth (int): Depth of the quantum circuits.
        lr (float): Learning rate.
        max_iters (int): Maximum number of iterations.
        noise_model (NoiseModel, optional): Noise model to apply. Defaults to None.

    Returns:
        dict: Dictionary containing comparison metrics and optimization history.
    """
    import time
    
    # Initialize quantum circuits for U and V
    circuit_U = create_parameterized_circuit(num_qubits, circuit_depth, 'U')
    circuit_V = create_parameterized_circuit(num_qubits, circuit_depth, 'V')
    
    # Initialize parameters randomly
    params = initialize_random_parameters(circuit_U, circuit_V)
    
    # Initialize backend
    backend = AerSimulator(method='statevector')
    
    loss_history = []
    singular_values_history = []
    
    start_time = time.time()
    
    # Add adaptive learning rate
    initial_lr = lr
    
    for iter in range(max_iters):
        # Compute loss and gradients
        current_loss = loss_function(matrix, circuit_U, circuit_V, 
                                   params[:len(circuit_U.parameters)],
                                   params[len(circuit_U.parameters):],
                                   rank, noise_model)
        
        # Compute gradients using parameter shift
        gradients = compute_gradients(circuit_U, circuit_V, params, matrix, rank)
        
        # Adaptive learning rate
        lr = initial_lr / (1 + iter/100)
        
        # Update parameters with gradient clipping
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > 1.0:
            gradients = gradients / grad_norm
        params = params - lr * gradients
        
        # Extract current singular values
        singular_values = extract_singular_values(circuit_U, circuit_V, params, matrix)
        
        # Store history
        loss_history.append(current_loss)
        singular_values_history.append(singular_values)
        
        # Early stopping check
        if len(loss_history) > 10:
            if np.abs(loss_history[-1] - loss_history[-10]) < 1e-6:
                print("Converged!")
                break
        
        if iter % 10 == 0:
            print(f"Iteration {iter}: Loss = {current_loss:.6f}")
            print(f"Singular Values: {singular_values}")
    
    training_time = time.time() - start_time
    
    # Extract final singular values
    final_singular_values = singular_values_history[-1]
    
    # Compare with classical SVD
    comparison = compare_with_classical_svd(matrix, final_singular_values, training_time, rank)
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


def initialize_random_parameters(circuit_U: QuantumCircuit, circuit_V: QuantumCircuit) -> np.ndarray:
    """
    Initialize the parameters for the quantum circuits U and V randomly.

    Args:
        circuit_U (QuantumCircuit): Quantum circuit for U.
        circuit_V (QuantumCircuit): Quantum circuit for V.

    Returns:
        np.ndarray: Array of initialized parameters, concatenated for U and V.
    """
    params_U = np.random.uniform(-np.pi, np.pi, len(circuit_U.parameters))
    params_V = np.random.uniform(-np.pi, np.pi, len(circuit_V.parameters))
    return np.concatenate([params_U, params_V])


def measure_quantum_expectation(circuit_U: QuantumCircuit, circuit_V: QuantumCircuit, 
                               params: np.ndarray, matrix: np.ndarray, backend: AerSimulator, noise_model: NoiseModel = None) -> float:
    """
    Measures the expectation value of the loss function for the current parameters.

    Args:
        circuit_U (QuantumCircuit): Quantum circuit for U.
        circuit_V (QuantumCircuit): Quantum circuit for V.
        params (np.ndarray): Array of parameters for U and V circuits.
        matrix (np.ndarray): The matrix to decompose.
        backend (AerSimulator): Qiskit Aer simulator backend.
        noise_model (NoiseModel, optional): Noise model to apply. Defaults to None.

    Returns:
        float: The measured expectation value.
    """
    # Split parameters for U and V
    num_params_U = len(circuit_U.parameters)
    params_U = params[:num_params_U]
    params_V = params[num_params_U:]
    
    # Bind parameters to circuits
    binding_U = {param: value for param, value in zip(circuit_U.parameters, params_U)}
    binding_V = {param: value for param, value in zip(circuit_V.parameters, params_V)}
    
    bound_circuit_U = circuit_U.assign_parameters(binding_U)
    bound_circuit_V = circuit_V.assign_parameters(binding_V)
    
    # Combine U and V into a single circuit
    combined_circuit = bound_circuit_U.compose(bound_circuit_V)
    
    # Measure the combined circuit
    combined_circuit.save_statevector()
    
    # Execute the circuit
    job = backend.run(combined_circuit, noise_model=noise_model)
    result = job.result()
    
    try:
        state = result.get_statevector()
    except QiskitError:
        print("Failed to retrieve statevector.")
        return float('inf')
    
    # Compute U * M * V†
    U = Operator(bound_circuit_U).data
    V = Operator(bound_circuit_V).data
    decomposed_matrix = U @ matrix @ V.conj().T
    
    # The expectation value could be the trace of decomposed_matrix
    expectation = np.abs(np.trace(decomposed_matrix))
    return expectation


def update_parameters(params: np.ndarray, gradients: np.ndarray, lr: float) -> np.ndarray:
    """
    Updates the parameters using gradient descent.

    Args:
        params (np.ndarray): Current parameters.
        gradients (np.ndarray): Gradients of the loss with respect to parameters.
        lr (float): Learning rate.

    Returns:
        np.ndarray: Updated parameters.
    """
    return params - lr * gradients


def extract_singular_values(circuit_U: QuantumCircuit, circuit_V: QuantumCircuit, 
                           params: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Extracts singular values from the quantum circuits U and V with given parameters.

    Args:
        circuit_U (QuantumCircuit): Quantum circuit for U.
        circuit_V (QuantumCircuit): Quantum circuit for V.
        params (np.ndarray): Array of parameters for U and V circuits.
        matrix (np.ndarray): The matrix to decompose.

    Returns:
        np.ndarray: Array of singular values.
    """
    # Split parameters for U and V
    num_params_U = len(circuit_U.parameters)
    params_U = params[:num_params_U]
    params_V = params[num_params_U:]
    
    # Bind parameters to circuits
    binding_U = {param: value for param, value in zip(circuit_U.parameters, params_U)}
    binding_V = {param: value for param, value in zip(circuit_V.parameters, params_V)}
    
    bound_circuit_U = circuit_U.assign_parameters(binding_U)
    bound_circuit_V = circuit_V.assign_parameters(binding_V)
    
    # Obtain unitary matrices
    U = Operator(bound_circuit_U).data
    V = Operator(bound_circuit_V).data
    
    # Compute U * M * V†
    decomposed_matrix = U @ matrix @ V.conj().T
    
    # Singular values are the absolute values of the diagonal elements in the ideal case
    singular_values = np.abs(np.diag(decomposed_matrix))
    
    return singular_values


def compare_with_classical_svd(matrix: np.ndarray, quantum_singular_values: np.ndarray, training_time: float, rank: int) -> dict:
    """
    Compares the quantum SVD results with classical SVD.

    Args:
        matrix (np.ndarray): Original matrix to decompose.
        quantum_singular_values (np.ndarray): Singular values obtained from QSVD.
        training_time (float): Time taken for the QSVD training.
        rank (int): Number of singular values/components to compare.

    Returns:
        dict: Dictionary containing comparison metrics.
    """
    # Perform classical SVD
    _, classical_singular_values, _ = classical_svd(matrix, full_matrices=False)
    classical_singular_values = classical_singular_values[:rank]
    
    # Normalize singular values for comparison
    quantum_singular_values_sorted = np.sort(quantum_singular_values)[::-1]
    classical_singular_values_sorted = np.sort(classical_singular_values)[::-1]
    
    # Compute accuracy metrics
    accuracy = np.mean(np.isclose(quantum_singular_values_sorted, classical_singular_values_sorted, atol=1e-2))
    difference = np.linalg.norm(quantum_singular_values_sorted - classical_singular_values_sorted)
    
    comparison = {
        "Classical Singular Values": classical_singular_values_sorted,
        "Quantum Singular Values": quantum_singular_values_sorted,
        "Accuracy": accuracy,
        "Difference": difference,
        "Training Time (s)": training_time
    }
    
    return comparison


def compute_gradients(circuit, params, matrix):
    gradients = []
    shift = np.pi/2
    
    for i in range(len(params)):
        # Shift parameter in positive direction
        params_plus = params.copy()
        params_plus[i] += shift
        
        # Shift parameter in negative direction
        params_minus = params.copy()
        params_minus[i] -= shift
        
        # Compute gradient using parameter-shift rule
        expectation_plus = measure_quantum_expectation(circuit, params_plus, matrix)
        expectation_minus = measure_quantum_expectation(circuit, params_minus, matrix)
        gradient = (expectation_plus - expectation_minus) / (2 * np.sin(shift))
        
        gradients.append(gradient)
    
    return np.array(gradients)



