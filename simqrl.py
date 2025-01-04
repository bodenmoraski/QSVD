import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.linalg import svd as classical_svd
from QSVD.simqsvd import SimulatedQSVD, PerformanceMonitor
from running_mean_std import RunningMeanStd
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit.quantum_info import partial_trace
from functools import partial
import random
from collections import deque
from torch.distributions import Normal
import logging
from logging_config import setup_logging

__all__ = [
    'SimulatedQuantumEnv',
    'QSVDAgent',
    'PPOAgent',
    'train_agent',
    'initialize_metrics',
    'plot_training_metrics',
    'AdaptiveLearningRate',
    'EarlyStopping',
    'CoherentError'
]

# Initialize logging
logger = setup_logging()

class CoherentError(Exception):
    """Exception raised for quantum coherence-related errors."""
    pass

def initialize_metrics():
    """
    Initialize comprehensive metrics dictionary for tracking QSVD performance
    
    Returns:
        dict: Dictionary containing all metric categories for training monitoring
    """
    return {
        # Basic training metrics
        'rewards': [],
        'errors': [],
        'fidelity_history': [],
        'loss_values': [],
        
        # SVD-specific metrics
        'svd_errors': [],
        'relative_errors': [],
        'singular_value_accuracy': [],
        
        # Quantum circuit metrics
        'circuit_quality': [],
        'noise_levels': [],
        'quantum_performance': {
            'circuit_depth': [],
            'gate_error_rates': [],
            'qubit_coherence': [],
            'quantum_volume': []
        },
        
        # Training stability metrics
        'gradient_stats': {
            'norms': [],
            'vanishing_events': 0,
            'exploding_events': 0
        },
        
        # Moving averages
        'moving_averages': {
            'reward': 0.0,
            'error': 0.0,
            'fidelity': 0.0
        },
        
        # Performance trends
        'trends': {
            'fidelity': 0.0,
            'error': 0.0,
            'learning_rate': []
        },
        
        # Convergence criteria
        'convergence_criteria': {
            'fidelity_threshold_met': False,
            'error_threshold_met': False,
            'stability_achieved': False
        },
        
        # Resource utilization
        'computational_metrics': {
            'circuit_execution_times': [],
            'memory_usage': [],
            'parameter_count': []
        },
        
        # Success tracking
        'success_rate': 0.0,
        'best_performance': {
            'fidelity': -float('inf'),
            'error': float('inf'),
            'epoch': 0
        },
        
        # Debug information
        'debug_events': {
            'nan_events': 0,
            'reset_events': 0,
            'error_events': 0
        }
    }

class SimulatedQuantumEnv:
    def __init__(self, num_qubits, circuit_depth, noise_params=None):
        logger.debug(f"Initializing SimulatedQuantumEnv with {num_qubits} qubits and depth {circuit_depth}")
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.noise_params = noise_params if noise_params else {}
        
        # Define state_size based on quantum state dimension (2^n for n qubits)
        self.state_size = 2**num_qubits
        logger.debug(f"State size set to: {self.state_size}")
        
        # Define action_size based on parameters needed for U and V circuits
        self.params_per_circuit = num_qubits * 3  # Example: RX, RY, RZ per qubit
        self.action_size = 2 * self.params_per_circuit  # Parameters for both U and V circuits
        logger.debug(f"Action size set to: {self.action_size}")
        
        # Initialize quantum state
        self.matrix = self.initialize_matrix()
        self.current_state = self.reset()
        
        logger.debug(f"SimulatedQuantumEnv initialized with matrix shape: {self.matrix.shape}")

    def reset(self):
        logger.debug("Resetting environment to initial state")
        # Existing reset code...
        logger.debug(f"Initial state: {self.current_state}")

    def step(self, action):
        try:
            logger.debug(f"Executing step with action: {action}")
            # Assuming action is a parameter vector for the circuit
            # Apply the action to the circuit
            self.apply_circuit_params(action, amp_noise=0.01, phase_noise=0.02)
            # Simulate the circuit execution
            result = self.simulate_circuit_execution()
            # Calculate the reward based on the result
            reward = self.calculate_reward(result)
            # Determine the next state based on the result
            next_state = self.get_next_state(result)
            logger.debug(f"Next state: {next_state}, Reward: {reward}")
        except CoherentError as ce:
            logger.error(f"Coherence error in step: {str(ce)}")
            # Handle coherence error
        except Exception as e:
            logger.error(f"Step failed with unexpected error: {str(e)}")
            # Handle other errors

    def apply_noise_to_circuit(self, u_params, v_params):
        logger.debug("Applying noise to circuit")
        # Existing noise application code...

    def apply_circuit_params(self, params, amp_noise, phase_noise):
        logger.debug(f"Applying circuit parameters with amp_noise={amp_noise}, phase_noise={phase_noise}")
        # Existing circuit parameter application code...

    def apply_gate(self, gate_matrix, qubit_index):
        logger.debug(f"Applying gate to qubit {qubit_index}")
        # Existing gate application code...

    def rotation_gate(self, gate_type, angle):
        logger.debug(f"Generating rotation gate: {gate_type} with angle {angle}")
        # Existing rotation gate code...

    def calculate_reward(self, result):
        logger.debug(f"Calculating reward based on result: {result}")
        # Existing reward calculation code...

    def get_next_state(self, result):
        logger.debug(f"Determining next state from result: {result}")
        # Existing next state determination code...

    def initialize_matrix(self):
        """Initialize the quantum state matrix.
        
        Returns:
            np.ndarray: Initial matrix representing the quantum state
        """
        try:
            # Initialize as identity matrix scaled by noise level (if specified)
            noise_level = self.noise_params.get('matrix_noise', 0.1)
            
            # Create identity matrix of size 2^num_qubits (dimension of quantum state space)
            matrix = np.identity(2**self.num_qubits, dtype=complex)
            
            # Apply initial noise if specified
            if noise_level > 0:
                matrix = matrix * (1 + noise_level)
                
            logger.debug(f"Initialized matrix with shape {matrix.shape}")
            return matrix
            
        except Exception as e:
            logger.error(f"Failed to initialize matrix: {str(e)}")
            raise CoherentError("Failed to initialize quantum matrix") from e

def train_agent(episodes, env):
    logger.info(f"Starting training for {episodes} episodes.")
    training_metrics = {
        'rewards': [],
        'errors': [],
        'fidelity_history': []
    }

    for episode in range(episodes):
        try:
            state = env.reset()
            logger.info(f"Episode {episode} started with initial state.")

            episode_reward = 0
            trajectories = []

            for step in range(env.circuit_depth):
                action = env.agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                trajectories.append((state, action, reward, next_state, done))
                training_metrics['rewards'].append(reward)
                training_metrics['fidelity_history'].append(env.calculate_reward(env.current_state))
                state = next_state
                episode_reward += reward

                if done:
                    break

            # After collecting trajectories, perform PPO update
            env.agent.update(trajectories)
            logger.info(f"Episode {episode} completed with reward: {episode_reward}")
        
        except CoherentError as ce:
            logger.error(f"Coherent error in episode {episode}: {str(ce)}")
            training_metrics['errors'].append(str(ce))
            continue
        except Exception as e:
            logger.error(f"Unexpected error in episode {episode}: {str(e)}")
            training_metrics['errors'].append(str(e))
            continue

    logger.info("Training completed.")
    return env.agent, training_metrics

def plot_training_metrics(metrics, save_path=None):
    """
    Plot training metrics including rewards, errors, and fidelity history.
    
    Args:
        metrics (dict): Dictionary containing training metrics
        save_path (str, optional): Path to save the plot. If None, displays plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot rewards
    plt.subplot(3, 1, 1)
    plt.plot(metrics['rewards'], label='Rewards')
    plt.title('Training Metrics')
    plt.ylabel('Reward')
    plt.legend()
    
    # Plot errors
    plt.subplot(3, 1, 2)
    plt.plot(metrics['fidelity_history'], label='Fidelity')
    plt.ylabel('Fidelity')
    plt.legend()
    
    # Plot fidelity
    plt.subplot(3, 1, 3)
    error_counts = [i for i in range(len(metrics['errors']))]
    plt.plot(error_counts, label='Errors')
    plt.xlabel('Episode')
    plt.ylabel('Error Count')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


