import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.linalg import svd as classical_svd
from simqsvd import SimulatedQSVD, PerformanceMonitor
from running_mean_std import RunningMeanStd
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit.quantum_info import partial_trace
from functools import partial
import random
from collections import deque
from torch.distributions import Normal
import logging
from logging_config import setup_logging
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, ReadoutError
from qiskit.quantum_info import state_fidelity

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
        """Initialize the quantum environment."""
        logger.debug("=== Initializing Quantum Environment ===")
        logger.debug(f"Number of qubits: {num_qubits}")
        logger.debug(f"Circuit depth: {circuit_depth}")
        logger.debug(f"Matrix size: {2**num_qubits}x{2**num_qubits}")
        
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.noise_params = noise_params if noise_params else {}
        
        # Calculate state size (real + imaginary parts of the matrix + time step)
        n = 2**num_qubits
        matrix_size = n * n  # Size of flattened matrix
        self.state_size = 2 * matrix_size + 1  # Real + Imaginary parts + time step
        
        logger.debug(f"State dimensions:")
        logger.debug(f"- Matrix size (n x n): {n} x {n}")
        logger.debug(f"- Flattened matrix size: {matrix_size}")
        logger.debug(f"- Total state size (with real/imag parts + time): {self.state_size}")
        
        # Parameters per circuit (3 params per qubit: Rx, Ry, Rz)
        self.params_per_circuit = 3 * num_qubits
        
        # Action size is parameters for both U and V circuits
        self.action_size = 2 * self.params_per_circuit
        
        logger.debug(f"Action dimensions:")
        logger.debug(f"- Parameters per circuit: {self.params_per_circuit}")
        logger.debug(f"- Total action size: {self.action_size}")
        
        # Episode tracking
        self.max_steps = circuit_depth
        self.current_step = 0
        
        # Initialize state
        self.matrix = self.initialize_matrix()
        self.current_state = None
        self.agent = None
        
        # Verify matrix dimensions
        if self.matrix.shape != (n, n):
            logger.error(f"Matrix shape mismatch! Expected ({n}, {n}), got {self.matrix.shape}")
            raise ValueError(f"Invalid matrix dimensions: {self.matrix.shape}")
            
        logger.debug("=== Environment Initialization Complete ===")
        logger.debug(f"Initial matrix shape: {self.matrix.shape}")
        logger.debug(f"Initial matrix norm: {np.linalg.norm(self.matrix)}")

    def initialize_matrix(self):
        """Initialize the quantum state matrix."""
        # Create a random complex matrix of appropriate size
        n = 2**self.num_qubits
        matrix = np.random.rand(n, n) + 1j * np.random.rand(n, n)
        
        # Normalize the matrix
        matrix = matrix / np.linalg.norm(matrix)
        
        logger.debug(f"Initialized matrix with shape {matrix.shape}")
        return matrix

    def reset(self):
        """Reset the environment to initial state"""
        logger.debug("Resetting environment to initial state")
        
        # Reset step counter
        self.current_step = 0
        
        # Initialize the matrix
        initial_matrix = self.initialize_matrix()
        
        # Prepare state observation
        self.current_state = self._prepare_state_observation(initial_matrix)
        
        logger.debug(f"Initial matrix shape: {initial_matrix.shape}")
        return self.current_state

    def set_agent(self, agent):
        """Set the agent for this environment"""
        self.agent = agent
        logger.info("Agent connected to environment")

    def step(self, action):
        """Execute one step in the environment"""
        try:
            logger.debug(f"Executing step with action: {action}")
            
            # Split action into U and V circuit parameters
            u_params = action[:self.params_per_circuit]
            v_params = action[self.params_per_circuit:]
            
            # Increment step counter
            self.current_step += 1
            
            # Simulate circuit
            next_state, fidelity = self.simulate_circuit_execution(u_params, v_params)
            
            if next_state is None:
                # If simulation failed, return current state with negative reward
                logger.warning("Circuit simulation failed, returning current state")
                return self.current_state, -1.0, True, {'fidelity': 0.0}
            
            # Update current state
            self.current_state = next_state
            
            # Calculate reward (modified to encourage exploration)
            reward = self.calculate_reward(fidelity)
            
            # Check if episode is done
            done = (self.current_step >= self.max_steps) or (fidelity > 0.99)
            
            info = {
                'fidelity': fidelity,
                'step': self.current_step
            }
            
            return next_state, reward, done, info
            
        except Exception as e:
            logger.error(f"Step failed with unexpected error: {str(e)}")
            # Return safe fallback values
            return self.current_state, -1.0, True, {'fidelity': 0.0}

    def simulate_circuit_execution(self, u_params, v_params):
        """
        Simulates the quantum circuit execution with proper error handling
        """
        try:
            logger.debug("=== Starting Circuit Execution ===")
            
            # Create quantum circuit for U
            qc_u = QuantumCircuit(self.num_qubits)
            logger.debug(f"Created U circuit with {self.num_qubits} qubits")
            
            # Add parameterized gates for U rotation
            for i in range(self.num_qubits):
                qc_u.rx(u_params[i*3], i)
                qc_u.ry(u_params[i*3 + 1], i)
                qc_u.rz(u_params[i*3 + 2], i)
            logger.debug(f"Added rotation gates to U circuit: {qc_u}")
            
            # Create quantum circuit for V
            qc_v = QuantumCircuit(self.num_qubits)
            
            # Add parameterized gates for V rotation
            for i in range(self.num_qubits):
                qc_v.rx(v_params[i*3], i)
                qc_v.ry(v_params[i*3 + 1], i)
                qc_v.rz(v_params[i*3 + 2], i)
            logger.debug(f"Added rotation gates to V circuit: {qc_v}")
            
            # Combine circuits
            qc = qc_u.compose(qc_v)
            qc.save_statevector()
            logger.debug(f"Combined circuit depth: {qc.depth()}")
            logger.debug(f"Circuit instructions: {qc.count_ops()}")
            
            # Set up backend with noise model
            backend = Aer.get_backend('aer_simulator')
            noise_model = self._create_noise_model()
            logger.debug("Created noise model")
            
            # Set up proper execution parameters
            execution_options = {
                'noise_model': noise_model,
                'basis_gates': ['rx', 'ry', 'rz'],
                'optimization_level': 1,
                'shots': 1024,
                'seed_simulator': np.random.randint(1000)
            }
            logger.debug(f"Execution options: {execution_options}")
            
            # Execute the circuit
            job = backend.run(qc, **execution_options)
            result = job.result()
            
            if not result.success:
                raise RuntimeError("Circuit execution failed")
                
            # Get statevector
            statevector = result.get_statevector(qc)
            if statevector is None:
                raise ValueError("No statevector returned")
                
            logger.debug(f"Statevector shape: {statevector.shape}")
            logger.debug(f"Statevector size: {len(statevector)}")
            
            # Calculate expected dimensions
            n = 2**self.num_qubits
            logger.debug(f"Expected statevector size: {n}")
            
            if len(statevector) != n:
                logger.error(f"Statevector size mismatch! Got {len(statevector)}, expected {n}")
                raise ValueError(f"Invalid statevector size: {len(statevector)}")
            
            # Calculate fidelity with respect to target state
            target_state = self._get_target_state()
            fidelity = state_fidelity(statevector, target_state)
            logger.debug(f"Calculated fidelity: {fidelity}")
            
            # Reshape statevector to matrix form for state observation
            state_matrix = statevector.reshape(int(np.sqrt(n)), int(np.sqrt(n)))
            logger.debug(f"Reshaped to matrix of shape {state_matrix.shape}")
            
            # Prepare next state observation
            next_state = self._prepare_state_observation(state_matrix)
            logger.debug(f"Prepared next state with shape {next_state.shape}")
            
            # Add noise reduction based on training progress
            noise_scale = max(0.1, 1.0 - (self.current_step / self.max_steps))
            fidelity = fidelity * (1.0 + 0.1 * (1.0 - noise_scale))
            logger.debug(f"Adjusted fidelity with noise scale {noise_scale}: {fidelity}")
            
            return next_state, fidelity
            
        except Exception as e:
            logger.error(f"Circuit execution error: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            # Return safe fallback values
            return None, 0.0
        
    def _create_noise_model(self):
        """Create a noise model for the quantum circuit"""
        from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, ReadoutError
        
        noise_model = NoiseModel()
        
        # Get noise parameters
        p_reset = self.noise_params.get('thermal_pop', 0.01)
        p_meas = self.noise_params.get('readout', {}).get('bit_flip', 0.015)
        
        # Add thermal relaxation error
        t1 = self.noise_params.get('t1', 50e-6)
        t2 = self.noise_params.get('t2', 70e-6)
        
        # Single-qubit gate times (in seconds)
        single_qubit_gates = ['rx', 'ry', 'rz']
        single_qubit_time = 20e-9  # 20 nanoseconds for single-qubit gates
        
        # Create thermal relaxation error for single-qubit gates
        single_qubit_error = thermal_relaxation_error(
            t1=t1,
            t2=t2,
            time=single_qubit_time
        )
        
        # Add single-qubit errors
        for gate in single_qubit_gates:
            noise_model.add_all_qubit_quantum_error(single_qubit_error, gate)
        
        # Create readout error for each qubit
        for qubit in range(self.num_qubits):
            # Create error probabilities for this qubit
            error_probs = np.array([[1 - p_meas, p_meas],
                                   [p_meas, 1 - p_meas]])
            
            # Create and add the readout error
            readout = ReadoutError(error_probs)
            noise_model.add_readout_error(readout, [qubit])
        
        return noise_model
        
    def _prepare_state_observation(self, matrix):
        """
        Converts matrix state to observation vector
        
        Args:
            matrix: Complex matrix of shape (2^n, 2^n) where n is num_qubits
            
        Returns:
            Observation vector containing:
            - Flattened real parts of matrix
            - Flattened imaginary parts of matrix
            - Normalized time step
        """
        # Ensure matrix is 2D numpy array
        matrix = np.asarray(matrix)
        if len(matrix.shape) == 1:
            n = int(np.sqrt(len(matrix)))
            matrix = matrix.reshape(n, n)
            
        # Get matrix dimensions
        n = 2**self.num_qubits
        expected_shape = (n, n)
        
        # If matrix is smaller than expected, pad it
        if matrix.shape[0] < n or matrix.shape[1] < n:
            padded = np.zeros(expected_shape, dtype=complex)
            padded[:matrix.shape[0], :matrix.shape[1]] = matrix
            matrix = padded
        # If matrix is larger than expected, truncate it
        elif matrix.shape[0] > n or matrix.shape[1] > n:
            matrix = matrix[:n, :n]
        
        # Ensure matrix is normalized
        matrix = matrix / np.linalg.norm(matrix)
        
        # Flatten matrix and get real/imaginary parts
        flattened = matrix.flatten()
        real_parts = np.real(flattened)
        imag_parts = np.imag(flattened)
        
        # Create observation vector
        observation = np.concatenate([
            real_parts,
            imag_parts,
            [self.current_step / self.max_steps]
        ])
        
        logger.debug(f"Matrix shape after processing: {matrix.shape}")
        logger.debug(f"Observation shape: {observation.shape}")
        logger.debug(f"Real parts shape: {real_parts.shape}, Imag parts shape: {imag_parts.shape}")
        
        return observation
        
    def _get_target_state(self):
        """
        Returns the target state for fidelity calculation
        
        For QSVD, we want a state that represents the ideal singular value decomposition
        of our input matrix. This should be a state of size 2^n where n is num_qubits.
        """
        logger.debug("=== Generating Target State ===")
        
        # Calculate classical SVD of the initial matrix for reference
        U, S, Vh = np.linalg.svd(self.matrix)
        logger.debug(f"Classical SVD shapes - U: {U.shape}, S: {S.shape}, Vh: {Vh.shape}")
        
        # For quantum state, we need a state vector of size 2^n
        n = 2**self.num_qubits
        logger.debug(f"Target state size should be {n}")
        
        # Create target state from the first singular vector
        # We use the first column of U as our target state
        target_state = U[:, 0]
        
        # Ensure it's the right size
        if len(target_state) != n:
            logger.warning(f"Target state size mismatch, reshaping from {len(target_state)} to {n}")
            target_state = np.resize(target_state, n)
        
        # Normalize the state
        target_state = target_state / np.linalg.norm(target_state)
        logger.debug(f"Created target state with shape {target_state.shape}")
        
        # Verify normalization
        norm = np.linalg.norm(target_state)
        logger.debug(f"Target state norm: {norm}")
        if not np.isclose(norm, 1.0):
            logger.warning(f"Target state not normalized! Norm = {norm}")
            target_state = target_state / norm
        
        return target_state

    def calculate_reward(self, fidelity):
        """
        Calculate reward based on fidelity and training progress
        
        Args:
            fidelity: Float between 0 and 1 representing quantum state fidelity
            
        Returns:
            Float: Calculated reward value
        """
        # Base reward from fidelity (shifted and scaled to [-1, 1])
        base_reward = 2.0 * fidelity - 1.0
        
        # Progressive reward scaling based on training progress
        progress = self.current_step / self.max_steps
        scale = 1.0 + progress  # Reward becomes more important later in training
        
        # Add bonus for high fidelity
        if fidelity > 0.9:
            bonus = 10.0 * (fidelity - 0.9)**2  # Quadratic bonus for high fidelity
        elif fidelity > 0.7:
            bonus = 2.0 * (fidelity - 0.7)  # Linear bonus for moderate fidelity
        else:
            bonus = 0.0
            
        # Add penalty for very low fidelity
        if fidelity < 0.3:
            penalty = -3.0 * (0.3 - fidelity)**2  # Quadratic penalty for very low fidelity
        else:
            penalty = 0.0
            
        # Combine all components
        reward = scale * base_reward + bonus + penalty
        
        logger.debug(f"Reward calculation:")
        logger.debug(f"- Base reward: {base_reward:.4f}")
        logger.debug(f"- Scale: {scale:.4f}")
        logger.debug(f"- Bonus: {bonus:.4f}")
        logger.debug(f"- Penalty: {penalty:.4f}")
        logger.debug(f"- Final reward: {reward:.4f}")
        
        return reward

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

def train_agent(episodes, env):
    """Train the agent in the environment"""
    training_metrics = {
        'rewards': [],
        'info_history': [],
        'episode_rewards': [],
        'episode_fidelities': []
    }
    
    try:
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_fidelity = 0
            trajectories = []
            
            for step in range(env.circuit_depth):
                action = env.agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                trajectories.append((state, action, reward, next_state, done))
                training_metrics['rewards'].append(reward)
                training_metrics['info_history'].append(info)
                
                episode_reward += reward
                episode_fidelity = info['fidelity']  # Keep track of final fidelity
                state = next_state
                
                if done:
                    break
            
            # Update agent after episode
            env.agent.update(trajectories)
            
            # Store episode-level metrics
            training_metrics['episode_rewards'].append(episode_reward)
            training_metrics['episode_fidelities'].append(episode_fidelity)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(training_metrics['episode_rewards'][-10:])
                avg_fidelity = np.mean(training_metrics['episode_fidelities'][-10:])
                logger.info(f"Episode {episode + 1}/{episodes}")
                logger.info(f"Average Reward: {avg_reward:.4f}")
                logger.info(f"Average Fidelity: {avg_fidelity:.4f}")
            
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise
        
    return env.agent, training_metrics

def plot_training_metrics(metrics, save_path=None):
    """
    Plot training metrics including rewards and fidelity history.
    
    Args:
        metrics (dict): Dictionary containing training metrics
        save_path (str, optional): Path to save the plot. If None, displays plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot rewards
    plt.subplot(2, 1, 1)
    plt.plot(metrics['rewards'], label='Rewards')
    plt.title('Training Metrics')
    plt.ylabel('Reward')
    plt.xlabel('Step')
    plt.legend()
    
    # Plot fidelity
    plt.subplot(2, 1, 2)
    fidelities = [info.get('fidelity', 0.0) for info in metrics.get('info_history', [])]
    plt.plot(fidelities, label='Fidelity')
    plt.ylabel('Fidelity')
    plt.xlabel('Step')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Bound actions to [-1, 1]
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single value output
        )
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # PPO specific parameters
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.ppo_update_iters = 5
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        # Action distribution
        self.action_std = 0.5
        
    def select_action(self, state):
        """Select action using the current policy"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action_mean = self.actor(state)
            dist = Normal(action_mean, self.action_std)
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            return action.squeeze().numpy()
            
    def update(self, trajectories):
        """Update policy using the PPO algorithm"""
        # Convert trajectories to tensors
        states = torch.FloatTensor([t[0] for t in trajectories])
        actions = torch.FloatTensor([t[1] for t in trajectories])
        rewards = torch.FloatTensor([t[2] for t in trajectories])
        next_states = torch.FloatTensor([t[3] for t in trajectories])
        dones = torch.FloatTensor([t[4] for t in trajectories])
        
        # Calculate advantages and returns
        with torch.no_grad():
            values = self.critic(states).squeeze(-1)  # Remove last dimension
            next_values = self.critic(next_states).squeeze(-1)  # Remove last dimension
            
            # GAE calculation
            advantages = torch.zeros_like(rewards)
            gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = next_values[t]
                else:
                    next_value = values[t + 1]
                    
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
                
            returns = advantages + values
            
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.ppo_update_iters):
            # Get action probabilities
            action_mean = self.actor(states)
            dist = Normal(action_mean, self.action_std)
            curr_probs = dist.log_prob(actions).sum(dim=-1)
            
            # Calculate ratio
            ratio = torch.exp(curr_probs)
            
            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            
            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (ensure dimensions match)
            value_pred = self.critic(states).squeeze(-1)  # Remove last dimension
            value_loss = F.mse_loss(value_pred, returns)
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': value_loss.item(),
            'advantages': advantages.mean().item()
        }


