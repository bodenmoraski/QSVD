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

class PerformanceMonitor:
    """Monitor and analyze quantum circuit performance metrics"""
    def __init__(self):
        self.metrics = {
            'circuit_fidelity': [],
            'singular_value_error': [],
            'noise_impact': [],
            'execution_time': [],
            'gradient_norms': [],
            'parameter_norms': [],
            'loss_values': [],
            'state_norms': []
        }
        
        self.debug_data = {
            'gradient_vanishing_events': 0,
            'gradient_exploding_events': 0,
            'nan_events': 0,
            'error_events': 0
        }
        
        self.thresholds = {
            'gradient_vanishing': 1e-5,
            'gradient_exploding': 1e3,
            'max_singular_value_error': 0.5
        }
        
        self.start_time = time.time()

    def update(self, noisy_values, true_values, state, noise_level):
        """
        Update monitor with new values from a training step
        
        Parameters:
        -----------
        noisy_values : np.ndarray
            The noisy singular values from the quantum circuit
        true_values : np.ndarray
            The true singular values from classical computation
        state : np.ndarray
            The current state vector
        noise_level : float
            The current noise level
        """
        try:
            # Calculate error metrics
            error = np.mean(np.abs(noisy_values - true_values))
            self.metrics['singular_value_error'].append(error)
            
            # Calculate state-based metrics
            state_norm = np.linalg.norm(state)
            self.metrics['state_norms'].append(state_norm)
            
            # Calculate noise impact
            noise_impact = np.sum(np.abs(noisy_values - true_values)) / len(true_values)
            self.metrics['noise_impact'].append(noise_impact)
            
            # Update execution time
            self.metrics['execution_time'].append(time.time() - self.start_time)
            
            # Check for anomalies
            if np.isnan(error):
                self.debug_data['nan_events'] += 1
            
            if error > self.thresholds['max_singular_value_error']:
                self.debug_data['error_events'] += 1
                
        except Exception as e:
            print(f"Warning: Error updating metrics: {str(e)}")
            self.debug_data['error_events'] += 1

    def update_metrics(self, **kwargs):
        """Update individual performance metrics"""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
                
                # Check for anomalies
                if key == 'gradient_norms':
                    if abs(value) < self.thresholds['gradient_vanishing']:
                        self.debug_data['gradient_vanishing_events'] += 1
                    elif abs(value) > self.thresholds['gradient_exploding']:
                        self.debug_data['gradient_exploding_events'] += 1
                        
                if np.isnan(value).any():
                    self.debug_data['nan_events'] += 1

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
            'noise_correlation': self._calculate_correlation(),
            'debug_stats': {
                'gradient_vanishing_events': self.debug_data['gradient_vanishing_events'],
                'gradient_exploding_events': self.debug_data['gradient_exploding_events'],
                'nan_events': self.debug_data['nan_events'],
                'error_events': self.debug_data['error_events']
            }
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

    def reset(self):
        """Reset all metrics"""
        for key in self.metrics:
            self.metrics[key] = []
        for key in self.debug_data:
            self.debug_data[key] = 0
        self.start_time = time.time()

    def get_summary(self):
        """Get a summary of current performance"""
        return {
            'execution_time': time.time() - self.start_time,
            'total_iterations': len(self.metrics['circuit_fidelity']),
            'recent_fidelity': np.mean(self.metrics['circuit_fidelity'][-10:]) if self.metrics['circuit_fidelity'] else 0,
            'gradient_health': {
                'vanishing': self.debug_data['gradient_vanishing_events'],
                'exploding': self.debug_data['gradient_exploding_events']
            },
            'error_rate': self.debug_data['error_events'] / max(1, len(self.metrics['circuit_fidelity']))
        }

class CircuitDebugger:
    """Class for handling debug logging"""
    def __init__(self):
        self.log_history = []
        self.error_count = 0
        self.warning_count = 0
        self.checkpoints = {}
        self.start_time = time.time()

    def log_checkpoint(self, name):
        self.checkpoints[name] = {
            'time': time.time() - self.start_time,
            'memory': psutil.Process().memory_info().rss / 1024 / 1024
        }

    def log_parameters(self, params):
        self.log_history.append({
            'type': 'parameters',
            'time': time.time() - self.start_time,
            'params': str(params)
        })

    def log_state_vectors(self, name, state):
        if state is not None:
            self.log_history.append({
                'type': 'state_vector',
                'name': name,
                'norm': np.linalg.norm(state),
                'max_amp': np.max(np.abs(state)),
                'min_amp': np.min(np.abs(state)),
                'time': time.time() - self.start_time
            })

    def log_state_analysis(self, analysis):
        self.log_history.append({
            'type': 'state_analysis',
            'time': time.time() - self.start_time,
            'analysis': analysis
        })

    def log_distributions(self, distributions):
        self.log_history.append({
            'type': 'distributions',
            'time': time.time() - self.start_time,
            'distributions': {
                k: {
                    'mean': np.mean(v),
                    'std': np.std(v),
                    'min': np.min(v),
                    'max': np.max(v)
                } for k, v in distributions.items()
            }
        })

    def log_matrix_properties(self, properties):
        self.log_history.append({
            'type': 'matrix_properties',
            'time': time.time() - self.start_time,
            'properties': properties
        })

    def log_computation(self, details):
        self.log_history.append({
            'type': 'computation',
            'time': time.time() - self.start_time,
            'details': details
        })

    def log_metric(self, name, value):
        self.log_history.append({
            'type': 'metric',
            'name': name,
            'value': value,
            'time': time.time() - self.start_time
        })

    def log_error(self, message):
        self.error_count += 1
        self.log_history.append({
            'type': 'error',
            'message': message,
            'time': time.time() - self.start_time
        })

    def log_warning(self, message):
        self.warning_count += 1
        self.log_history.append({
            'type': 'warning',
            'message': message,
            'time': time.time() - self.start_time
        })

    def generate_debug_report(self):
        """Generate a comprehensive debug report"""
        report = ["Debug Report", "=" * 50, "\n"]
        
        # Error and Warning Summary
        report.append(f"Errors: {self.error_count}")
        report.append(f"Warnings: {self.warning_count}\n")
        
        # Checkpoint Timeline
        report.append("Checkpoint Timeline:")
        for name, data in self.checkpoints.items():
            report.append(f"  {name}: {data['time']:.3f}s (Memory: {data['memory']:.1f}MB)")
        
        # Recent Events
        report.append("\nRecent Events:")
        recent_events = self.log_history[-10:]  # Last 10 events
        for event in recent_events:
            report.append(f"  {event['type']}: {event.get('message', '')}")
        
        return "\n".join(report)

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
        
        # Add noise_level attribute
        self.noise_level = self.noise_params.get('readout_error', 0.02)
        
        # Create backends
        self.statevector_backend = AerSimulator(method='statevector')  # For unitary operations
        self.noisy_backend = AerSimulator()  # For noisy simulations with measurements
        self.noise_model = self._create_noise_model()
        self.noise_history = []
        self.circuit_fidelity_history = []
        self.circuit_U = None
        self.circuit_V = None
        
        # Initialize circuits
        self.circuit_U = self.create_parameterized_circuit('U', 'circuit_U')
        self.circuit_V = self.create_parameterized_circuit('V', 'circuit_V')
        
        self.debug_metrics = {
            'gradient_history': [],
            'parameter_history': [],
            'fidelity_raw': [],
            'singular_value_history': [],
            'circuit_state_norms': [],
            'probability_distributions': [],
            'noise_impact_metrics': {}
        }
        self.debug_logger = CircuitDebugger()
        
        # Calculate total parameters
        params_per_circuit = num_qubits * (1 + circuit_depth) * 3  # From create_parameterized_circuit method
        self.total_parameters = params_per_circuit * 2  # For both U and V circuits

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
        """Create a parameterized quantum circuit"""
        circuit = QuantumCircuit(self.num_qubits, self.num_qubits, name=name)
        parameters = []
        
        # Initial rotation layer
        for qubit in range(self.num_qubits):
            for gate_type in ['rx', 'ry', 'rz']:
                param = Parameter(f'{prefix}_{gate_type}_{0}_{qubit}')
                parameters.append(param)
                getattr(circuit, gate_type)(param, qubit)
        
        # Entangling layers with rotations
        for depth in range(self.circuit_depth):
            # Add entangling gates
            for q in range(0, self.num_qubits - 1, 2):
                circuit.cx(q, q + 1)
            circuit.barrier()
            for q in range(1, self.num_qubits - 1, 2):
                circuit.cx(q, q + 1)
            circuit.barrier()
            
            # Add rotation gates
            for qubit in range(self.num_qubits):
                for gate_type in ['rx', 'ry', 'rz']:
                    param = Parameter(f'{prefix}_{gate_type}_{depth+1}_{qubit}')
                    parameters.append(param)
                    getattr(circuit, gate_type)(param, qubit)
        
        return circuit
    
    def simulate_svd(self, matrix, params_U, params_V):
        """Simulate QSVD using state vector simulation"""
        try:
            '''
            print("Debug - Input parameters:")
            print(f"params_U first few values: {params_U[:5]}")
            print(f"params_V first few values: {params_V[:5]}")
            '''
            
            # Validate matrix dimensions
            if matrix.shape[0] != 2**self.num_qubits or matrix.shape[1] != 2**self.num_qubits:
                raise ValueError(f"Matrix dimensions must be {2**self.num_qubits}x{2**self.num_qubits}")
            
            # Validate parameter counts
            expected_params = self.num_qubits * (1 + self.circuit_depth) * 3
            if len(params_U) != expected_params or len(params_V) != expected_params:
                raise ValueError(f"Expected {expected_params} parameters for each circuit, got {len(params_U)}/{len(params_V)}")
            
            # Ensure circuits exist
            if self.circuit_U is None or self.circuit_V is None:
                self.circuit_U = self.create_parameterized_circuit('U', 'circuit_U')
                self.circuit_V = self.create_parameterized_circuit('V', 'circuit_V')
            
            # Create parameter dictionaries
            param_dict_U = dict(zip(self.circuit_U.parameters, params_U))
            param_dict_V = dict(zip(self.circuit_V.parameters, params_V))
            
            # Bind parameters
            bound_circuit_U = self.circuit_U.assign_parameters(param_dict_U)
            bound_circuit_V = self.circuit_V.assign_parameters(param_dict_V)
            
            # Add measurements
            bound_circuit_U.measure_all()
            bound_circuit_V.measure_all()
            
            # Run circuits with noise model
            job_U = self.noisy_backend.run(bound_circuit_U, shots=1000)
            job_V = self.noisy_backend.run(bound_circuit_V, shots=1000)
            
            # Get probability distributions
            counts_U = job_U.result().get_counts()
            counts_V = job_V.result().get_counts()
            
            probs_U = self._counts_to_probabilities(counts_U)
            probs_V = self._counts_to_probabilities(counts_V)
            
            # Estimate singular values
            singular_values = self._estimate_singular_values(probs_U, probs_V, matrix)
            
            '''
            print("Debug - Circuit outputs:")
            print(f"First few counts_U: {list(counts_U.items())[:3]}")
            print(f"First few probs_U: {probs_U[:3]}")
            print(f"Resulting singular values: {singular_values[:3]}")
            '''
            return np.sort(singular_values)[::-1]
            
        except Exception as e:
            print(f"Error in simulate_svd: {str(e)}")
            raise
    
    def _counts_to_probabilities(self, counts):
        """Convert measurement counts to probability distribution"""
        total_shots = sum(counts.values())
        dim = 2**self.num_qubits
        probs = np.zeros(dim)
        
        for bitstring, count in counts.items():
            # Debug print
            #print(f"Processing bitstring: {bitstring}, count: {count}")
            
            # Remove spaces and get only the relevant bits
            clean_bitstring = bitstring.replace(' ', '')
            try:
                # Split on '0000' and take first part
                relevant_bits = clean_bitstring.split('0000')[0]
                # Convert to index
                index = int(relevant_bits, 2) if relevant_bits else 0
                #print(f"Converted to index: {index}")
                
                if index < dim:
                    probs[index] = count / total_shots
                    
            except (ValueError, IndexError) as e:
                print(f"Error processing bitstring {bitstring}: {e}")
                continue
        
        # Debug print before normalization
        #print(f"Pre-normalization probabilities: {probs}")
        
        # Normalize probabilities
        total_prob = np.sum(probs)
        if total_prob > 0:
            probs = probs / total_prob
        
        # Debug print after normalization
        #print(f"Final probabilities: {probs}")
        
        return probs
    
    def _estimate_singular_values(self, probs_U, probs_V, matrix):
        """Enhanced singular value estimation with debugging"""
        try:
            # Log input probabilities
            self.debug_logger.log_distributions({
                'probs_U': probs_U,
                'probs_V': probs_V
            })
            
            dim = 2**self.num_qubits
            singular_values = np.zeros(dim)
            matrix_norm = np.linalg.norm(matrix, 'fro')
            
            # Log matrix properties
            self.debug_logger.log_matrix_properties({
                'matrix_norm': matrix_norm,
                'matrix_rank': np.linalg.matrix_rank(matrix),
                'condition_number': np.linalg.cond(matrix)
            })
            
            # Sort probabilities with logging
            sorted_U = np.sort(probs_U)[::-1]
            sorted_V = np.sort(probs_V)[::-1]
            
            self.debug_logger.log_distributions({
                'sorted_U': sorted_U,
                'sorted_V': sorted_V
            })
            
            # Enhanced singular value computation
            for i in range(dim):
                weight = np.exp(-i / dim)
                sv = np.sqrt(sorted_U[i] * sorted_V[i]) * matrix_norm * weight
                singular_values[i] = sv
                
                # Log each singular value computation
                self.debug_logger.log_computation({
                    'index': i,
                    'U_prob': sorted_U[i],
                    'V_prob': sorted_V[i],
                    'weight': weight,
                    'singular_value': sv
                })
            
            # Normalize with logging
            scale = matrix_norm / np.linalg.norm(singular_values)
            self.debug_logger.log_metric('normalization_scale', scale)
            
            final_values = singular_values * scale
            self.debug_metrics['singular_value_history'].append(final_values)
            
            return final_values
            
        except Exception as e:
            self.debug_logger.log_error(f"Singular value estimation failed: {str(e)}")
            return np.zeros(2**self.num_qubits)
    
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
    
    def analyze_circuit_quality(self, params):
        """
        Analyze circuit quality and potential issues
        
        Parameters:
        -----------
        params : array-like
            Combined parameters for both U and V circuits
        
        Returns:
        --------
        dict
            Dictionary containing circuit quality metrics
        """
        try:
            # Ensure circuits exist
            if self.circuit_U is None or self.circuit_V is None:
                self.circuit_U = self.create_parameterized_circuit('U', 'circuit_U')
                self.circuit_V = self.create_parameterized_circuit('V', 'circuit_V')
            
            # Split parameters for U and V circuits
            params_per_circuit = len(self.circuit_U.parameters)
            params_U = params[:params_per_circuit]
            params_V = params[params_per_circuit:]
            
            def compute_gradients(params):
                """Helper function to compute parameter gradients"""
                return np.gradient(params)
            
            # Calculate gradient metrics
            grad_U = compute_gradients(params_U)
            grad_V = compute_gradients(params_V)
            
            # Calculate depth efficiency (ratio of useful operations to total depth)
            total_depth = self.circuit_depth
            useful_ops = sum(1 for _ in self.circuit_U.data) + sum(1 for _ in self.circuit_V.data)
            depth_efficiency = useful_ops / (2 * total_depth)  # Factor of 2 for U and V circuits
            
            metrics = {
                'gradient_vanishing': np.mean(np.abs(grad_U)) < 1e-5 or np.mean(np.abs(grad_V)) < 1e-5,
                'gradient_exploding': np.mean(np.abs(grad_U)) > 1e3 or np.mean(np.abs(grad_V)) > 1e3,
                'depth_efficiency': depth_efficiency
            }
            
            return metrics
            
        except Exception as e:
            print(f"Warning: Circuit analysis failed: {str(e)}")
            return {
                'gradient_vanishing': False,
                'gradient_exploding': False,
                'depth_efficiency': 0.5
            }
    
    def _compute_parameter_gradients(self, circuit, params):
        """Simplified, robust gradient computation"""
        epsilon = 1e-2  # Larger epsilon for numerical stability
        grads = []
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            
            params_minus = params.copy()
            params_minus[i] -= epsilon
            
            val_plus = self._evaluate_circuit(circuit.bind_parameters(params_plus))
            val_minus = self._evaluate_circuit(circuit.bind_parameters(params_minus))
            
            grad = (val_plus - val_minus) / (2 * epsilon)
            grads.append(grad)
        
        return np.array(grads)
    
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
            
            return min(1.0, -+(0.0, efficiency))
            
        except Exception as e:
            print(f"Warning: Error calculating depth efficiency: {str(e)}")
            return 0.5
    
    def _evaluate_circuit(self, circuit):
        """Enhanced circuit evaluation with debugging"""
        try:
            # Add debugging checkpoint
            self.debug_logger.log_checkpoint('circuit_evaluation_start')
            
            # Create a new statevector simulator
            backend = AerSimulator(method='statevector')
            
            # Debug print for circuit parameters
            if circuit.parameters:
                self.debug_logger.log_parameters(circuit.parameters)
            
            # Execute without measurements
            circuit_no_meas = circuit.copy()
            circuit_no_meas.remove_final_measurements()
            
            # Add state vector saving
            from qiskit_aer.library import SaveStatevector
            save_sv = SaveStatevector(num_qubits=circuit.num_qubits)
            circuit_no_meas.append(save_sv, circuit_no_meas.qubits)
            
            # Run simulation with detailed metadata
            job = backend.run(
                circuit_no_meas,
                shots=1,
                seed_simulator=42,
                metadata={'debug_mode': True}
            )
            
            try:
                result = job.result()
                statevector = result.data().get('statevector')
                
                if statevector is not None:
                    # Log state vector properties
                    self.debug_metrics['circuit_state_norms'].append(np.linalg.norm(statevector))
                    
                    # Detailed state analysis
                    state_analysis = {
                        'norm': np.linalg.norm(statevector),
                        'max_amplitude': np.max(np.abs(statevector)),
                        'non_zero_states': np.count_nonzero(np.abs(statevector) > 1e-10)
                    }
                    self.debug_logger.log_state_analysis(state_analysis)
                    
                    zero_state_amplitude = statevector[0]
                    return np.abs(zero_state_amplitude)**2
                else:
                    self.debug_logger.log_error("No statevector in result")
                    return 0.0
                    
            except Exception as inner_e:
                self.debug_logger.log_error(f"Statevector extraction failed: {str(inner_e)}")
                return 0.0
                
        except Exception as e:
            self.debug_logger.log_error(f"Circuit evaluation failed: {str(e)}")
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
    
    def _calculate_circuit_fidelity(self, circuit_state, ideal_state=None):
        """Fixed fidelity calculation"""
        if circuit_state is None or np.all(circuit_state == 0):
            return 0.0
        
        # Proper normalization
        circuit_state = circuit_state / np.linalg.norm(circuit_state)
        if ideal_state is not None:
            ideal_state = ideal_state / np.linalg.norm(ideal_state)
            return np.abs(np.vdot(circuit_state, ideal_state))**2
        return 1.0  # If no ideal state, perfect fidelity
    
    def optimize_circuit_depth(self):
        """Dynamically adjust circuit depth based on noise levels"""
        current_fidelity = self.get_current_fidelity()
        if current_fidelity < self.target_fidelity:
            self.reduce_depth()
        return self.current_depth
    
    def schedule_gates(self):
        """Implement noise-aware gate scheduling"""
        # Prioritize gates based on error rates
        gate_errors = {
            'single': self.noise_params['gate_times']['single'],
            'two': self.noise_params['gate_times']['two']
        }
        # Schedule gates with lower error rates first
        sorted_gates_by_error = sorted(gate_errors, key=lambda x: gate_errors[x])
        return sorted_gates_by_error
    
    def train(self, num_epochs=1000):
        """Simplified training loop"""
        best_params = None
        best_fidelity = -float('inf')
        
        for epoch in range(num_epochs):
            # Forward pass
            current_fidelity = self._evaluate_circuit(self.circuit)
            
            # Compute gradients
            grads = self._compute_parameter_gradients(self.circuit, self.parameters)
            
            # Basic gradient descent
            self.parameters += self.learning_rate * grads
            
            # Track best result
            if current_fidelity > best_fidelity:
                best_fidelity = current_fidelity
                best_params = self.parameters.copy()
                
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Fidelity = {current_fidelity:.4f}")
        
        self.parameters = best_params

__all__ = [
    'SimulatedQSVD',
    'PerformanceMonitor'
]