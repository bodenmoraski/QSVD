# Implementing QGD for a cost function, not currently working, to be added

import numpy as np
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.utils import QuantumInstance
from typing import List, Tuple, Optional, Union, Dict
from .dynamicepisolon import Quantum_GD, cost_function, qgd_steps, run_qc, visualize_circuit_evolution

class QuantumGradientDescent:
    """
    A class implementing Quantum Gradient Descent optimization for quantum circuits.
    """
    
    def __init__(self, 
                 n_qubits: int,
                 target_state: List[complex],
                 learning_rate: float = 0.1,
                 epsilon: float = 0.01,
                 max_iterations: int = 100):
        """
        Initialize the QGD optimizer.
        
        Args:
            n_qubits: Number of qubits in the circuit
            target_state: Target quantum state to achieve
            learning_rate: Learning rate for gradient descent
            epsilon: Small value for numerical gradient calculation
            max_iterations: Maximum number of iterations
        """
        self.n_qubits = n_qubits
        self.target_state = target_state
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.simulator = Aer.get_backend('statevector_simulator')
        self.parameters = []
        self.cost_history = []
        
    def create_parameterized_circuit(self) -> QuantumCircuit:
        """
        Creates a parameterized quantum circuit.
        
        Returns:
            QuantumCircuit: Parameterized quantum circuit
        """
        qc = QuantumCircuit(self.n_qubits)
        
        # Create parameters for rotation gates
        self.parameters = [Parameter(f'θ_{i}') for i in range(self.n_qubits * 3)]
        
        # Add parameterized rotation gates for each qubit
        for i in range(self.n_qubits):
            qc.rx(self.parameters[i*3], i)
            qc.ry(self.parameters[i*3 + 1], i)
            qc.rz(self.parameters[i*3 + 2], i)
            
        return qc
    
    def compute_cost(self, statevector: np.ndarray) -> float:
        """
        Compute the cost between current state and target state.
        
        Args:
            statevector: Current quantum state vector
            
        Returns:
            float: Cost value
        """
        return sum(abs(target - actual) ** 2 
                  for target, actual in zip(self.target_state, statevector))
    
    def compute_gradient(self, 
                        circuit: QuantumCircuit, 
                        param_values: List[float]) -> np.ndarray:
        """
        Compute the gradient for all parameters using finite differences.
        
        Args:
            circuit: Quantum circuit
            param_values: Current parameter values
            
        Returns:
            np.ndarray: Gradient vector
        """
        gradients = []
        
        for i, param in enumerate(self.parameters):
            # Forward evaluation
            forward_params = param_values.copy()
            forward_params[i] += self.epsilon
            
            forward_circuit = circuit.bind_parameters({
                param: value for param, value in zip(self.parameters, forward_params)
            })
            
            forward_result = self.simulator.run(transpile(forward_circuit, 
                                                        self.simulator)).result()
            forward_statevector = forward_result.get_statevector()
            forward_cost = self.compute_cost(forward_statevector)
            
            # Backward evaluation
            backward_params = param_values.copy()
            backward_params[i] -= self.epsilon
            
            backward_circuit = circuit.bind_parameters({
                param: value for param, value in zip(self.parameters, backward_params)
            })
            
            backward_result = self.simulator.run(transpile(backward_circuit, 
                                                         self.simulator)).result()
            backward_statevector = backward_result.get_statevector()
            backward_cost = self.compute_cost(backward_statevector)
            
            # Central difference gradient
            gradient = (forward_cost - backward_cost) / (2 * self.epsilon)
            gradients.append(gradient)
            
        return np.array(gradients)
    
    def optimize(self, initial_params: Optional[List[float]] = None) -> Tuple[List[float], float]:
        """
        Perform quantum gradient descent optimization.
        
        Args:
            initial_params: Initial parameter values (optional)
            
        Returns:
            Tuple[List[float], float]: Optimized parameters and final cost
        """
        circuit = self.create_parameterized_circuit()
        
        if initial_params is None:
            current_params = np.random.uniform(0, 2*np.pi, len(self.parameters))
        else:
            current_params = initial_params
            
        self.cost_history = []
        
        for iteration in range(self.max_iterations):
            # Bind current parameters
            bound_circuit = circuit.bind_parameters({
                param: value for param, value in zip(self.parameters, current_params)
            })
            
            # Evaluate current cost
            result = self.simulator.run(transpile(bound_circuit, self.simulator)).result()
            statevector = result.get_statevector()
            current_cost = self.compute_cost(statevector)
            self.cost_history.append(current_cost)
            
            # Check convergence
            if current_cost < 1e-6:
                break
                
            # Compute gradients and update parameters
            gradients = self.compute_gradient(circuit, current_params)
            current_params = current_params - self.learning_rate * gradients
            
        return current_params, current_cost
    
    def get_cost_history(self) -> List[float]:
        """
        Returns the history of cost values during optimization.
        
        Returns:
            List[float]: Cost history
        """
        return self.cost_history



