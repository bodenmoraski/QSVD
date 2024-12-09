import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, pauli_error, depolarizing_error
from scipy.linalg import svd as classical_svd
from qiskit.quantum_info.operators import Pauli
from qiskit.circuit.library import IGate, XGate, YGate, ZGate
import time
import psutil
import qiskit

class SimulatedQSVD:
    """
    Simulated Quantum SVD with realistic noise models using Qiskit
    """
    def __init__(self, num_qubits, circuit_depth, noise_params=None):
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        # Default noise parameters
        self.noise_params = noise_params or {
            't1': 50e-6,                # T1 relaxation time
            't2': 70e-6,               # T2 dephasing time
            'gate_times': {            # Gate operation times
                'single': 20e-9,       # Single-qubit gate time
                'two': 40e-9,          # Two-qubit gate time
            },
            'thermal_population': 0.01,
            'readout_error': 0.02,
            'crosstalk_strength': 0.03,
            'control_amplitude_error': 0.01,
            'control_frequency_drift': 0.005,
        }
        
        # Create backends
        self.statevector_backend = AerSimulator(method='statevector')  # For unitary operations
        self.noisy_backend = AerSimulator()  # For noisy simulations with measurements
        self.noise_model = self._create_noise_model()
        self.noise_history = []
        self.circuit_fidelity_history = []
        self.circuit_U = None
        self.circuit_V = None
        
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
        """Create a parameterized quantum circuit for U or V"""
        circuit = QuantumCircuit(self.num_qubits, name=name)
        parameters = []
        
        # Add parameterized gates
        for depth in range(self.circuit_depth):
            for qubit in range(self.num_qubits):
                param = Parameter(f'{prefix}_theta_{depth}_{qubit}')
                parameters.append(param)
                circuit.ry(param, qubit)
            
            # Add entangling gates
            if depth < self.circuit_depth - 1:
                for qubit in range(self.num_qubits - 1):
                    circuit.cx(qubit, qubit + 1)
        
        self.num_parameters = len(parameters)
        return circuit
    
    def simulate_svd(self, matrix, params_U, params_V):
        """Simulate QSVD using state vector simulation"""
        tracker = ErrorTracker()
        
        try:
            with tracker:
                # Create circuits and store them as instance variables
                self.circuit_U = self.create_parameterized_circuit('U')
                tracker.log_stage('circuit_creation_U')
                
                self.circuit_V = self.create_parameterized_circuit('V')
                tracker.log_stage('circuit_creation_V')
                
                # Create two versions of each circuit:
                # One for statevector simulation (no measurements)
                sv_circuit_U = self.circuit_U.copy()
                sv_circuit_V = self.circuit_V.copy()
                
                # One for noisy simulation (with measurements)
                meas_circuit_U = self.circuit_U.copy()
                meas_circuit_V = self.circuit_V.copy()
                meas_circuit_U.measure_all()
                meas_circuit_V.measure_all()
                
                # Bind parameters
                param_dict_U = dict(zip(self.circuit_U.parameters, params_U))
                param_dict_V = dict(zip(self.circuit_V.parameters, params_V))
                
                # Run statevector simulation for gradient calculations
                bound_sv_U = sv_circuit_U.assign_parameters(param_dict_U)
                bound_sv_V = sv_circuit_V.assign_parameters(param_dict_V)
                
                # Run noisy simulation with measurements
                bound_meas_U = meas_circuit_U.assign_parameters(param_dict_U)
                bound_meas_V = meas_circuit_V.assign_parameters(param_dict_V)
                
                # Get noisy measurement results
                noise_model, noise_impacts = self.create_advanced_noise_model()
                
                job_U = self.noisy_backend.run(
                    bound_meas_U,
                    noise_model=noise_model,
                    shots=1000,
                    optimization_level=0
                )
                
                job_V = self.noisy_backend.run(
                    bound_meas_V,
                    noise_model=noise_model,
                    shots=1000,
                    optimization_level=0
                )
                
                # Get probability distributions from measurements
                counts_U = job_U.result().get_counts()
                counts_V = job_V.result().get_counts()
                
                # Convert to probability distributions
                probs_U = self._counts_to_probabilities(counts_U)
                probs_V = self._counts_to_probabilities(counts_V)
                
                # Estimate singular values from probability distributions
                singular_values = self._estimate_singular_values(probs_U, probs_V, matrix)
                
                # Log noise impacts if needed
                self.noise_history.append(noise_impacts)
                
                return np.sort(singular_values)[::-1]
                
        except Exception as e:
            print(tracker.generate_error_report())
            raise
    
    def _counts_to_probabilities(self, counts):
        """Convert measurement counts to probability distribution"""
        total_shots = sum(counts.values())
        probs = np.zeros(2**self.num_qubits)
        
        for bitstring, count in counts.items():
            index = int(bitstring, 2)
            probs[index] = count / total_shots
        
        return probs
    
    def _estimate_singular_values(self, probs_U, probs_V, matrix):
        """Estimate singular values from probability distributions"""
        # Simple estimation method - can be improved
        dim = 2**self.num_qubits
        singular_values = np.zeros(dim)
        
        # Use probability distributions to estimate singular values
        for i in range(dim):
            singular_values[i] = np.sqrt(probs_U[i] * probs_V[i]) * np.linalg.norm(matrix)
        
        return singular_values
    
    def get_true_singular_values(self, matrix):
        """Get classical SVD for comparison"""
        _, s, _ = classical_svd(matrix)
        return s
    
    def create_advanced_noise_model(self, params=None):
        """Enhanced noise model with debugging capabilities"""
        params = params or self.noise_params
        noise_model = NoiseModel()
        noise_impacts = {}
        
        try:
            # Create thermal relaxation error
            t1, t2 = params['t1'], params['t2']
            gate_time = params['gate_times']['single']
            
            # Simplified thermal relaxation error
            error = thermal_relaxation_error(
                t1, t2, gate_time,
                excited_state_population=params['thermal_population']
            )
            
            # Add error to noise model
            noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
            noise_impacts['thermal_relaxation'] = params['thermal_population']
            
            # Add readout error
            readout_error = [
                [1 - params['readout_error'], params['readout_error']],
                [params['readout_error'], 1 - params['readout_error']]
            ]
            noise_model.add_all_qubit_readout_error(readout_error)
            noise_impacts['readout'] = params['readout_error']
            
            print("Noise Model Analysis:")
            for name, impact in noise_impacts.items():
                print(f"  {name}: {impact:.4f} impact on fidelity")
            
            return noise_model, noise_impacts
            
        except Exception as e:
            print(f"Warning: Error creating noise model: {str(e)}")
            # Return minimal noise model
            return NoiseModel(), {'base_error': 0.01}
    
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
    
    def analyze_circuit_quality(self, circuit_U, circuit_V, params):
        """
        Analyze circuit quality and potential issues
        """
        metrics = {}
        
        # Check parameter gradients
        params_U = params[:len(circuit_U.parameters)]
        params_V = params[len(circuit_U.parameters):]
        
        # Calculate gradient metrics
        grad_U = self._compute_parameter_gradients(circuit_U, params_U)
        grad_V = self._compute_parameter_gradients(circuit_V, params_V)
        
        # Check for vanishing/exploding gradients
        metrics['gradient_vanishing'] = np.mean(np.abs(grad_U)) < 1e-5 or np.mean(np.abs(grad_V)) < 1e-5
        metrics['gradient_exploding'] = np.mean(np.abs(grad_U)) > 1e3 or np.mean(np.abs(grad_V)) > 1e3
        
        # Calculate depth efficiency
        metrics['depth_efficiency'] = self._calculate_depth_efficiency(circuit_U, circuit_V)
        
        return metrics
    
    def _compute_parameter_gradients(self, circuit, params):
        """
        Compute parameter gradients for a circuit
        """
        try:
            # Simple finite difference approximation
            epsilon = 1e-7
            gradients = []
            
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += epsilon
                
                params_minus = params.copy()
                params_minus[i] -= epsilon
                
                # Bind parameters and compute difference
                param_dict_plus = dict(zip(circuit.parameters, params_plus))
                param_dict_minus = dict(zip(circuit.parameters, params_minus))
                
                circuit_plus = circuit.assign_parameters(param_dict_plus)
                circuit_minus = circuit.assign_parameters(param_dict_minus)
                
                # Compute gradient using finite difference
                grad = (self._evaluate_circuit(circuit_plus) - 
                       self._evaluate_circuit(circuit_minus)) / (2 * epsilon)
                
                gradients.append(grad)
            
            return np.array(gradients)
            
        except Exception as e:
            print(f"Warning: Error computing gradients: {str(e)}")
            return np.zeros(len(params))
    
    def _calculate_depth_efficiency(self, circuit_U, circuit_V):
        """
        Calculate circuit depth efficiency metric
        """
        try:
            # Simple metric based on circuit depth and number of parameters
            total_depth = circuit_U.depth() + circuit_V.depth()
            total_params = len(circuit_U.parameters) + len(circuit_V.parameters)
            
            # Efficiency = 1 / (normalized_depth * params_per_layer)
            efficiency = 1.0 / (total_depth * (total_params / total_depth))
            
            return min(1.0, max(0.0, efficiency))
            
        except Exception as e:
            print(f"Warning: Error calculating depth efficiency: {str(e)}")
            return 0.5
    
    def _evaluate_circuit(self, circuit):
        """Evaluate circuit output for gradient calculation using statevector"""
        try:
            # Ensure all parameters are bound
            if circuit.parameters:
                print(f"Warning: Circuit has unbound parameters: {circuit.parameters}")
                return 0.0
            
            # Create a new statevector simulator
            backend = AerSimulator(method='statevector')
            
            # Execute without measurements
            circuit_no_meas = circuit.copy()
            circuit_no_meas.remove_final_measurements()
            
            # Add save instruction with the correct number of qubits
            from qiskit_aer.library import SaveStatevector
            save_sv = SaveStatevector(num_qubits=circuit.num_qubits)  # Specify num_qubits
            circuit_no_meas.append(save_sv, circuit_no_meas.qubits)
            
            # Run simulation
            job = backend.run(
                circuit_no_meas,
                shots=1,
                seed_simulator=42  # For reproducibility
            )
            
            try:
                result = job.result()
                # Get statevector
                statevector = result.data().get('statevector')
                
                if statevector is not None:
                    zero_state_amplitude = statevector[0]
                    return np.abs(zero_state_amplitude)**2
                else:
                    print("Warning: No statevector found in result")
                    print(f"Available data keys: {list(result.data().keys())}")
                    return 0.0
                
            except Exception as inner_e:
                print(f"Warning: Error getting statevector: {str(inner_e)}")
                if 'result' in locals():
                    print(f"Available data keys: {list(result.data().keys())}")
                return 0.0
            
        except Exception as e:
            print(f"Warning: Error evaluating circuit: {str(e)}")
            print(f"Circuit details: {circuit}")
            return 0.0
    
    def _calculate_theoretical_impact(self, error_model):
        """
        Calculate theoretical impact of an error model on circuit fidelity
        
        Parameters:
        -----------
        error_model : QuantumError
            The quantum error model to analyze
        
        Returns:
        --------
        float
            Estimated impact on circuit fidelity (0 to 1, where 0 means no impact)
        """
        try:  
            # Get the Kraus operators from the error model
            kraus_ops = error_model.to_instruction().kraus_ops
            
            # Calculate the average fidelity impact
            dim = 2**self.num_qubits
            fidelity_impact = 0
            
            # For each Kraus operator
            for K in kraus_ops:
                # Calculate trace of K^†K
                trace = np.trace(K.conjugate().T @ K)
                fidelity_impact += np.abs(trace)**2
            
            # Normalize the impact
            fidelity_impact = 1 - (fidelity_impact / (dim**2))
            
            return fidelity_impact
            
        except Exception as e:
            print(f"Warning: Error calculating theoretical impact: {str(e)}")
            # Return a default impact value
            return 0.01

class ErrorTracker:
    def __init__(self):
        self.stages = []
        self.errors = []
        self.timings = {}
        self.start_time = None
    
    def __enter__(self):
        """Support for context manager protocol"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for context manager protocol"""
        self.end_time = time.time()
        if exc_type is not None:
            self.log_error(exc_val)
        return False  # Don't suppress exceptions
    
    def log_stage(self, stage_name):
        """Log a stage in the computation"""
        self.stages.append({
            'name': stage_name,
            'time': time.time(),
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024  # MB
        })
    
    def log_error(self, error):
        """Log an error"""
        self.errors.append({
            'time': time.time(),
            'error': str(error),
            'type': type(error).__name__
        })
    
    def generate_error_report(self):
        """Generate a comprehensive error report"""
        report = ["Error Report", "=" * 50]
        
        if self.errors:
            for i, error in enumerate(self.errors, 1):
                report.append(f"\nError {i}:")
                report.append(f"Type: {error['type']}")
                report.append(f"Message: {error['error']}")
                report.append(f"Time: {error['time'] - self.start_time:.2f}s after start")
        
        if self.stages:
            report.append("\nStage Timeline:")
            for stage in self.stages:
                report.append(f"- {stage['name']}: {stage['time'] - self.start_time:.2f}s")
                report.append(f"  Memory Usage: {stage['memory_usage']:.1f} MB")
        
        return "\n".join(report)

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'singular_value_error': [],
            'circuit_fidelity': [],
            'noise_impact': [],
            'optimization_progress': []
        }
    
    def update(self, current_values, true_values, circuit_state, noise_level):
        error = np.mean(np.abs(current_values - true_values))
        self.metrics['singular_value_error'].append(error)
        
        fidelity = self._calculate_circuit_fidelity(circuit_state)
        self.metrics['circuit_fidelity'].append(fidelity)
        
        self.metrics['noise_impact'].append(noise_level)
    
    def _calculate_circuit_fidelity(self, circuit_state):
        """
        Calculate quantum state fidelity from circuit state
        
        Parameters:
        -----------
        circuit_state : numpy.ndarray
            The quantum state vector or density matrix
        
        Returns:
        --------
        float
            Fidelity metric between 0 and 1
        """
        try:
            # If circuit_state is a state vector
            if isinstance(circuit_state, np.ndarray) and circuit_state.ndim == 1:
                # Calculate state purity
                return np.abs(np.vdot(circuit_state, circuit_state))
            
            # If circuit_state is a density matrix
            elif isinstance(circuit_state, np.ndarray) and circuit_state.ndim == 2:
                # Calculate trace fidelity
                return np.real(np.trace(circuit_state @ circuit_state.conj().T))
            
            # If circuit_state is a list of measurement outcomes
            elif isinstance(circuit_state, (list, np.ndarray)):
                # Calculate classical fidelity based on measurement statistics
                return 1.0 - np.mean(np.abs(circuit_state))
            
            else:
                print(f"Warning: Unexpected circuit state type: {type(circuit_state)}")
                return 0.5
                
        except Exception as e:
            print(f"Warning: Error calculating circuit fidelity: {str(e)}")
            return 0.5
    
    def generate_report(self):
        """Generate a report of the current metrics"""
        # Check if we have enough data points
        if len(self.metrics['circuit_fidelity']) < 2:
            return {
                'avg_error': np.mean(self.metrics['singular_value_error']) if self.metrics['singular_value_error'] else 0.0,
                'fidelity_trend': 0.0,  # Not enough data for gradient
                'noise_correlation': 0.0  # Not enough data for correlation
            }
            
        return {
            'avg_error': np.mean(self.metrics['singular_value_error']),
            'fidelity_trend': self._calculate_trend(self.metrics['circuit_fidelity']),
            'noise_correlation': self._calculate_correlation()
        }
    
    def _calculate_trend(self, data):
        """Calculate trend with safety checks"""
        if len(data) < 2:
            return 0.0
        try:
            # Calculate simple difference for trend
            # Positive value means improving, negative means degrading
            recent_data = data[-10:]  # Look at recent history
            return np.mean(np.diff(recent_data))
        except Exception as e:
            print(f"Warning: Error calculating trend: {str(e)}")
            return 0.0
    
    def _calculate_correlation(self):
        """Calculate correlation with safety checks"""
        try:
            if len(self.metrics['noise_impact']) < 2:
                return 0.0
            return np.corrcoef(
                self.metrics['noise_impact'], 
                self.metrics['singular_value_error']
            )[0,1]
        except Exception as e:
            print(f"Warning: Error calculating correlation: {str(e)}")
            return 0.0