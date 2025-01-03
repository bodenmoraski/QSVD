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
from scipy.linalg import expm
from scipy.stats import truncnorm

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
        self.matrix_dim = 2**num_qubits
        self.circuit_depth = circuit_depth
        
        # Comprehensive noise parameters with realistic defaults
        self.noise_params = noise_params or {
            # Decoherence parameters
            't1': 50e-6,              # T1 relaxation time (50 microseconds)
            't2': 70e-6,              # T2 dephasing time (70 microseconds)
            'thermal_pop': 0.01,      # Thermal excitation probability
            
            # Gate-level errors
            'single_gate': {
                'amplitude': 0.001,    # Amplitude damping
                'phase': 0.002,       # Phase damping
                'over_rotation': 0.003 # Over-rotation error
            },
            'two_gate': {
                'amplitude': 0.002,    # Two-qubit amplitude error
                'phase': 0.004,       # Two-qubit phase error
                'cross_talk': 0.003   # Cross-talk between qubits
            },
            
            # Measurement errors
            'readout': {
                'bit_flip': 0.015,    # Bit flip probability
                'thermal_noise': 0.01, # Thermal noise in readout
                'crosstalk': 0.008    # Readout crosstalk
            },
            
            # Environmental effects
            'magnetic_field': {
                'drift': 0.001,       # Magnetic field drift
                'fluctuation': 0.002  # Field fluctuations
            },
            'temperature': {
                'baseline': 0.02,     # Base temperature effects
                'fluctuation': 0.005  # Temperature fluctuations
            },
            
            # Quantum-specific effects
            'zz_coupling': 0.004,     # ZZ coupling strength
            'leakage': 0.002,         # Leakage to higher states
            'coherent_errors': 0.003  # Systematic coherent errors
        }
        
        # Initialize noise correlation matrix
        self.setup_noise_correlations()
        
    def setup_noise_correlations(self):
        """Setup spatial and temporal noise correlations"""
        n = self.matrix_dim
        # Spatial correlations between qubits
        self.spatial_correlations = np.exp(-np.abs(np.subtract.outer(range(n), range(n))) / 2)
        # Temporal correlation decay
        self.temporal_decay = np.exp(-np.arange(self.circuit_depth) / 5)
        
    def _generate_coherent_noise(self, shape):
        """Generate coherent noise with proper correlations"""
        base_noise = np.random.normal(0, 1, shape)
        return np.dot(self.spatial_correlations, base_noise)
        
    def _apply_decoherence(self, matrix, time):
        """Apply realistic decoherence effects"""
        t1, t2 = self.noise_params['t1'], self.noise_params['t2']
        
        # Amplitude damping
        gamma = 1 - np.exp(-time / t1)
        amp_damping = np.sqrt(1 - gamma) * matrix
        
        # Phase damping
        lambda_phase = 1 - np.exp(-time / t2)
        phase_damping = np.exp(-lambda_phase) * matrix
        
        return amp_damping * phase_damping
        
    def _add_measurement_noise(self, values):
        """Enhanced measurement noise simulation"""
        readout = self.noise_params['readout']
        
        # Generate base noise
        noise = np.random.normal(0, readout['thermal_noise'], values.shape)
        
        # Add bit-flip errors
        bit_flips = np.random.binomial(1, readout['bit_flip'], values.shape)
        values = np.where(bit_flips, -values, values)
        
        # Add readout crosstalk
        crosstalk_matrix = np.exp(-np.abs(np.subtract.outer(range(len(values)), range(len(values)))) * readout['crosstalk'])
        noise = np.dot(crosstalk_matrix, noise)
        
        return values + noise
        
    def _add_unitary_noise(self, matrix, params):
        """Advanced unitary noise simulation"""
        # Get relevant parameters
        single_gate = self.noise_params['single_gate']
        two_gate = self.noise_params['two_gate']
        
        # Parameter-dependent noise
        param_noise = np.mean(np.abs(params)) * single_gate['amplitude']
        
        # Generate coherent and incoherent noise components
        coherent_noise = self._generate_coherent_noise(matrix.shape) * self.noise_params['coherent_errors']
        incoherent_noise = np.random.normal(0, param_noise, matrix.shape)
        
        # Apply over-rotation errors
        over_rotation = single_gate['over_rotation'] * np.sin(2 * np.pi * params.mean())
        rotation_matrix = expm(1j * over_rotation * np.eye(matrix.shape[0]))
        
        # Apply ZZ coupling effects
        zz_coupling = self.noise_params['zz_coupling']
        coupling_matrix = np.exp(1j * zz_coupling * np.outer(params, params))
        
        # Combine all noise effects
        noisy_matrix = matrix @ rotation_matrix @ coupling_matrix
        noisy_matrix += coherent_noise + incoherent_noise
        
        # Apply decoherence
        gate_time = self.circuit_depth * (single_gate['amplitude'] + two_gate['amplitude'])
        noisy_matrix = self._apply_decoherence(noisy_matrix, gate_time)
        
        return noisy_matrix
        
    def _apply_environmental_effects(self, matrix):
        """Apply environmental noise effects"""
        mag_field = self.noise_params['magnetic_field']
        temp = self.noise_params['temperature']
        
        # Magnetic field effects
        field_noise = np.random.normal(mag_field['drift'], mag_field['fluctuation'])
        field_effect = expm(1j * field_noise * np.diag(np.random.randn(matrix.shape[0])))
        
        # Temperature effects
        temp_noise = truncnorm.rvs(-2, 2, loc=temp['baseline'], scale=temp['fluctuation'])
        temp_effect = np.exp(-temp_noise * np.abs(matrix))
        
        return matrix @ field_effect * temp_effect
        
    def simulate_svd(self, matrix, params_U, params_V):
        """
        Simulate QSVD output with comprehensive noise effects
        
        Args:
            matrix: Input matrix
            params_U, params_V: Circuit parameters
        """
        # Get classical SVD as baseline
        U, S, V = np.linalg.svd(matrix)
        
        # Apply comprehensive noise effects
        noisy_U = self._add_unitary_noise(U, params_U)
        noisy_U = self._apply_environmental_effects(noisy_U)
        
        noisy_S = self._add_measurement_noise(S)
        
        noisy_V = self._add_unitary_noise(V, params_V)
        noisy_V = self._apply_environmental_effects(noisy_V)
        
        # Apply leakage effects
        leakage = self.noise_params['leakage']
        if np.random.random() < leakage:
            # Simulate leakage to higher states
            leakage_factor = 1 - leakage
            noisy_U *= leakage_factor
            noisy_V *= leakage_factor
        
        return noisy_U, noisy_S, noisy_V

    def reset(self, matrix=None):
        """Reset the simulator state"""
        if matrix is None:
            # Generate a new random matrix if none provided
            matrix = np.random.rand(self.matrix_dim, self.matrix_dim) + \
                    1j * np.random.rand(self.matrix_dim, self.matrix_dim)
        
        # Reset internal state
        self.matrix = matrix
        self.setup_noise_correlations()
        print(f"generated Random matrix: {str(matrix)}")
        return matrix

__all__ = [
    'SimulatedQSVD',
    'PerformanceMonitor'
]