import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, pauli_error, depolarizing_error
from scipy.linalg import svd as classical_svd
from qiskit.quantum_info.operators import Pauli
from qiskit.circuit.library import IGate, XGate, YGate, ZGate

class SimulatedQSVD:
    """
    Simulated Quantum SVD with realistic noise models using Qiskit
    """
    def __init__(self, num_qubits, circuit_depth):
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        # Create two different backends for different purposes
        self.unitary_backend = AerSimulator(method='unitary')  # For perfect calculations
        self.noisy_backend = AerSimulator()                    # For noisy simulations
        self.noise_model = self._create_noise_model()
        
    def _create_noise_model(self, t1=50e-6, t2=70e-6, gate_time=20e-9):
        """Create a realistic noise model using Qiskit"""
        noise_model = NoiseModel()
        
        # Typical T1 and T2 times for superconducting qubits
        t1, t2 = 50e-6, 70e-6  # 50 and 70 microseconds
        gate_time = 20e-9      # 20 nanoseconds
        
        # Create relaxation error for single-qubit gates
        single_qubit_error = thermal_relaxation_error(t1, t2, gate_time)
        
        # Add errors to noise model
        noise_model.add_all_qubit_quantum_error(single_qubit_error, ['rx', 'ry', 'rz'])
        noise_model.add_all_qubit_quantum_error(single_qubit_error.tensor(single_qubit_error), ['cx'])
        
        return noise_model
    
    def apply_noise(self, values):
        """Simulate quantum noise on singular values. This is a placeholder for a more sophisticated noise model."""
        noise = np.random.normal(0, 0.01, values.shape)  # Assuming noise_level=0.01
        return values + noise
    
    def create_parameterized_circuit(self, prefix='', name=''):
        """Create a more expressive parameterized quantum circuit"""
        circuit = QuantumCircuit(self.num_qubits, name=name)
        parameters = []
        
        # Initial rotation layer
        for qubit in range(self.num_qubits):     
            for gate_type in ['rx', 'ry', 'rz']:
                param = Parameter(f'{prefix}_{gate_type}_{0}_{qubit}')
                parameters.append(param)
                getattr(circuit, gate_type)(param, qubit)
        
        # Entangling layers
        for depth in range(self.circuit_depth):
            # Entangling layer
            for qubit in range(0, self.num_qubits - 1, 2):
                circuit.cx(qubit, qubit + 1)
            circuit.barrier()
            for qubit in range(1, self.num_qubits - 1, 2):
                circuit.cx(qubit, qubit + 1)
            
            # Rotation layer
            for qubit in range(self.num_qubits):
                for gate_type in ['rx', 'ry', 'rz']:
                    param = Parameter(f'{prefix}_{gate_type}_{depth+1}_{qubit}')
                    parameters.append(param)
                    getattr(circuit, gate_type)(param, qubit)
        
        self.num_parameters = len(parameters)
        return circuit
    
    def simulate_svd(self, matrix, params_U, params_V):
        """Simulate QSVD using state vector simulation"""
        # Create circuits
        circuit_U = self.create_parameterized_circuit('U')
        circuit_V = self.create_parameterized_circuit('V')
        
        # Ensure parameters are the correct length
        expected_params = self.num_qubits * (1 + self.circuit_depth) * 3  # 3 gates per qubit per layer
        if len(params_U) != expected_params or len(params_V) != expected_params:
            raise ValueError(f"Expected {expected_params} parameters, got {len(params_U)}/{len(params_V)}")
        
        # Create parameter dictionaries
        param_dict_U = dict(zip([param for param in circuit_U.parameters], params_U))
        param_dict_V = dict(zip([param for param in circuit_V.parameters], params_V))
        
        # Bind parameters
        bound_circuit_U = circuit_U.assign_parameters(param_dict_U)
        bound_circuit_V = circuit_V.assign_parameters(param_dict_V)
        
        # Add measurements
        bound_circuit_U.measure_all()
        bound_circuit_V.measure_all()
        
        # Run circuits with noise model
        noise_model = self.create_advanced_noise_model()
        
        try:
            job_U = self.noisy_backend.run(
                bound_circuit_U,
                noise_model=noise_model,
                shots=1000,
                optimization_level=0
            )
            
            job_V = self.noisy_backend.run(
                bound_circuit_V,
                noise_model=noise_model,
                shots=1000,
                optimization_level=0
            )
            
            # Get probability distributions from measurements
            counts_U = job_U.result().get_counts()
            counts_V = job_V.result().get_counts()
            
        except Exception as e:
            print(f"Circuit execution failed: {str(e)}")
            # Return fallback values
            return np.sort(np.abs(np.diagonal(matrix)))[::-1]
        
        # Convert to probability distributions
        probs_U = self._counts_to_probabilities(counts_U)
        probs_V = self._counts_to_probabilities(counts_V)
        
        # Estimate singular values from probability distributions
        singular_values = self._estimate_singular_values(probs_U, probs_V, matrix)
        
        return np.sort(singular_values)[::-1]
    
    def _counts_to_probabilities(self, counts):
        """Convert measurement counts to probability distribution"""
        total_shots = sum(counts.values())
        probs = np.zeros(2**self.num_qubits)
        
        for bitstring, count in counts.items():
            index = int(bitstring, 2)
            probs[index] = count / total_shots
        
        return probs
    
    def _estimate_singular_values(self, probs_U, probs_V, matrix):
        """Improved singular value estimation"""
        dim = 2**self.num_qubits
        singular_values = np.zeros(dim)
        matrix_norm = np.linalg.norm(matrix, 'fro')  # Frobenius norm
        
        # Sort probabilities by magnitude
        sorted_U = np.sort(probs_U)[::-1]
        sorted_V = np.sort(probs_V)[::-1]
        
        # Use correlation between probabilities
        for i in range(dim):
            # Weight by position in sorted list
            weight = np.exp(-i / dim)  # Exponential decay weight
            singular_values[i] = np.sqrt(sorted_U[i] * sorted_V[i]) * matrix_norm * weight
        
        # Normalize to preserve matrix norm
        scale = matrix_norm / np.linalg.norm(singular_values)
        return singular_values * scale
    
    def get_true_singular_values(self, matrix):
        """Get classical SVD for comparison"""
        _, s, _ = classical_svd(matrix)
        return s
    
    def create_advanced_noise_model(self, 
                                  t1=50e-6,
                                  t2=70e-6,
                                  gate_times={
                                      'single': 20e-9,
                                      'two': 40e-9,
                                  },
                                  thermal_population=0.01,
                                  readout_error=0.02,
                                  crosstalk_strength=0.03,
                                  control_amplitude_error=0.01,
                                  control_frequency_drift=0.005,
                                  ):
        noise_model = NoiseModel()
        
        # Combine all single-qubit gate errors into one error model
        single_qubit_relax = thermal_relaxation_error(
            t1, t2, gate_times['single'], thermal_population
        )
        amp_damp = depolarizing_error(control_amplitude_error, 1)
        phase_error = pauli_error([
            ('Z', control_frequency_drift), 
            ('I', 1 - control_frequency_drift)
        ])
        
        # Combine all single-qubit errors into one
        single_qubit_error = single_qubit_relax.compose(amp_damp).compose(phase_error)
        
        # Add combined error once for all single-qubit gates
        noise_model.add_all_qubit_quantum_error(
            single_qubit_error, 
            ['rx', 'ry', 'rz', 'h']
        )
        
        # Two-qubit gate errors
        two_qubit_relax = thermal_relaxation_error(
            t1, t2, gate_times['two'], thermal_population
        ).tensor(thermal_relaxation_error(t1, t2, gate_times['two'], thermal_population))
        
        # Crosstalk error
        total_error_prob = crosstalk_strength
        individual_error_prob = total_error_prob / 6
        no_error_prob = 1.0 - total_error_prob
        
        crosstalk_error = pauli_error([
            ('IX', individual_error_prob),
            ('IY', individual_error_prob),
            ('IZ', individual_error_prob),
            ('XI', individual_error_prob),
            ('YI', individual_error_prob),
            ('ZI', individual_error_prob),
            ('II', no_error_prob)
        ])
        
        # Combine two-qubit errors
        two_qubit_error = two_qubit_relax.compose(crosstalk_error)
        noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx', 'cz'])
        
        # Add readout error
        readout_error_prob = [[1 - readout_error, readout_error],
                             [readout_error, 1 - readout_error]]
        noise_model.add_all_qubit_readout_error(readout_error_prob)
        
        return noise_model
    
    def apply_advanced_noise(self, circuit, noise_model):
        """
        Apply the advanced noise model to a quantum circuit
        
        Parameters:
        -----------
        circuit: QuantumCircuit
            The quantum circuit to apply noise to
        noise_model: NoiseModel
            The noise model to apply
        
        Returns:
        --------
        noisy_circuit: QuantumCircuit
            The circuit with noise applied
        """
        # Create a quantum circuit with the same structure
        noisy_circuit = circuit.copy()
        
        # Add barriers to prevent optimization removing noise effects
        noisy_circuit.barrier()
        
        # Add noise channels between gates
        for i in range(len(circuit.data) - 1):
            # Add decoherence effects between gates
            noisy_circuit.barrier()
            
            # Add random control errors
            if np.random.random() < 0.1:  # 10% chance of control error
                qubit = np.random.randint(0, circuit.num_qubits)
                angle = np.random.normal(0, 0.1)  # Small random angle
                noisy_circuit.rz(angle, qubit)
        
        return noisy_circuit