import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.linalg import svd as classical_svd
from simqsvd import SimulatedQSVD, PerformanceMonitor
from running_mean_std import RunningMeanStd
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit.quantum_info import partial_trace
from functools import partial
import random
from collections import deque
from torch.distributions import Normal

__all__ = [
    'SimulatedQuantumEnv',
    'QSVDAgent',
    'PPOAgent',
    'train_agent',
    'initialize_metrics',
    'plot_training_metrics',
    'AdaptiveLearningRate',
    'EarlyStopping'
]

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

class QSVDAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(QSVDAgent, self).__init__()
        
        # Policy Network
        self.policy_fc1 = nn.Linear(state_size, 256)
        self.policy_ln1 = nn.LayerNorm(256)
        self.policy_fc2 = nn.Linear(256, 256)
        self.policy_ln2 = nn.LayerNorm(256)
        self.policy_fc3 = nn.Linear(256, action_size)
        
        # Value Network
        self.value_fc1 = nn.Linear(state_size, 256)
        self.value_ln1 = nn.LayerNorm(256)
        self.value_fc2 = nn.Linear(256, 256)
        self.value_ln2 = nn.LayerNorm(256)
        self.value_fc3 = nn.Linear(256, 1)
        
        # Initialize weights
        for layer in [self.policy_fc1, self.policy_fc2, self.policy_fc3,
                      self.value_fc1, self.value_fc2, self.value_fc3]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        
        # Policy parameters
        self.log_std = nn.Parameter(torch.zeros(action_size))
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        
        # Replay buffer
        self.buffer = []
        
    def forward_policy(self, state):
        x = F.relu(self.policy_ln1(self.policy_fc1(state)))
        x = F.relu(self.policy_ln2(self.policy_fc2(x)))
        mean = torch.tanh(self.policy_fc3(x)) * np.pi
        std = torch.clamp(self.log_std.exp(), min=1e-6, max=1.0)
        return mean, std
    
    def forward_value(self, state):
        x = F.relu(self.value_ln1(self.value_fc1(state)))
        x = F.relu(self.value_ln2(self.value_fc2(x)))
        value = self.value_fc3(x)
        return value
    
    def get_action(self, state):
        """Get action from the current policy with proper state validation"""
        # Validate state
        if state is None:
            print("Warning: Received None state, using zero state")
            state = np.zeros(self.state_size)
        
        # Convert numpy array or list to tensor
        if isinstance(state, (np.ndarray, list)):
            state = torch.FloatTensor(state)
        elif not isinstance(state, torch.Tensor):
            raise TypeError(f"State must be numpy array, list, or tensor. Got {type(state)}")
        
        # Ensure state has correct shape
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
            
        with torch.no_grad():
            try:
                mean, std = self.forward_policy(state)
                
                # Safety checks
                if torch.isnan(mean).any() or torch.isnan(std).any():
                    print("Warning: NaN detected in action selection")
                    return np.zeros(mean.shape[1])  # Return zeros with correct shape
                
                dist = Normal(mean, std)
                action = dist.sample()
                return action.squeeze().detach().numpy()
            except Exception as e:
                print(f"Error sampling action: {e}")
                return np.zeros(self.action_size)
    
    def select_action(self, state):
        """Full action selection with log probability and value"""
        action = self.get_action(state)
        state = torch.FloatTensor(state)
        mean, std = self.forward_policy(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(torch.FloatTensor(action)).sum(dim=-1)
        value = self.forward_value(state)
        return action, log_prob.detach().numpy(), value.detach().numpy()
    
    def store_experience(self, state, action, log_prob, reward, next_state, done, value):
        self.buffer.append((state, action, log_prob, reward, next_state, done, value))
    
    def update(self, gamma=0.99, tau=0.95, clip_epsilon=0.2, epochs=10, batch_size=64):
        states, actions, old_log_probs, rewards, next_states, dones, values = zip(*self.buffer)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        values = torch.FloatTensor(values).squeeze()
        with torch.no_grad():
            next_values = self.forward_value(next_states).squeeze()
        
        # Compute GAE and returns
        advantages, returns = compute_gae(rewards, values.numpy().tolist() + [next_values.numpy()], dones)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        
        # PPO update
        for _ in range(epochs):
            for idx in range(0, len(states), batch_size):
                batch_states = states[idx:idx+batch_size]
                batch_actions = actions[idx:idx+batch_size]
                batch_old_log_probs = old_log_probs[idx:idx+batch_size]
                batch_returns = returns[idx:idx+batch_size]
                batch_advantages = advantages[idx:idx+batch_size]
                
                # Forward pass
                mean, std = self.forward_policy(batch_states)
                dist = Normal(mean, std)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                
                # Ratio for PPO clipping
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO Surrogate Loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value = self.forward_value(batch_states).squeeze()
                value_loss = F.mse_loss(value, batch_returns)
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                self.optimizer.step()
        
        # Clear the buffer after update
        self.buffer.clear()

class GradientMonitor:
    """Monitor and analyze gradient behavior during training"""
    def __init__(self):
        self.history = []
        self.vanishing_threshold = 1e-4  # Threshold for vanishing gradients
        self.exploding_threshold = 1e2   # Threshold for exploding gradients
        self.window_size = 50            # Size of sliding window for analysis
        
        # Metrics storage
        self.metrics = {
            'gradient_norms': [],
            'vanishing_events': 0,
            'exploding_events': 0,
            'stable_gradients': 0
        }
        
    def analyze_gradient(self, grad):
        """Analyze gradient statistics and detect issues"""
        if not isinstance(grad, np.ndarray):
            grad = np.array(grad)
            
        # Calculate basic statistics
        grad_norm = np.linalg.norm(grad)
        stats = {
            'norm': grad_norm,
            'mean': np.mean(grad),
            'std': np.std(grad),
            'max': np.max(np.abs(grad)),
            'min': np.min(np.abs(grad[np.nonzero(grad)])) if np.any(grad) else 0,
            'vanishing': grad_norm < self.vanishing_threshold,
            'exploding': grad_norm > self.exploding_threshold
        }
        
        # Update metrics
        self.metrics['gradient_norms'].append(grad_norm)
        if stats['vanishing']:
            self.metrics['vanishing_events'] += 1
        elif stats['exploding']:
            self.metrics['exploding_events'] += 1
        else:
            self.metrics['stable_gradients'] += 1
            
        # Store in history
        self.history.append(stats)
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        return stats
    
    def get_trend(self):
        """Analyze gradient trend over recent history"""
        if len(self.history) < 2:
            return 'insufficient_data'
            
        recent_norms = [h['norm'] for h in self.history[-10:]]
        trend = np.mean(np.diff(recent_norms))
        
        if abs(trend) < 1e-6:
            return 'stable'
        elif trend > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def get_summary(self):
        """Get summary of gradient behavior"""
        if not self.history:
            return "No gradient data available"
            
        total_updates = sum(self.metrics.values())
        if total_updates == 0:
            return "No updates recorded"
            
        return {
            'recent_trend': self.get_trend(),
            'vanishing_ratio': self.metrics['vanishing_events'] / total_updates,
            'exploding_ratio': self.metrics['exploding_events'] / total_updates,
            'stable_ratio': self.metrics['stable_gradients'] / total_updates,
            'recent_norm_mean': np.mean([h['norm'] for h in self.history[-10:]]),
            'recent_norm_std': np.std([h['norm'] for h in self.history[-10:]])
        }

class QSVDEnvironment:
    def __init__(self, num_qubits=4, circuit_depth=4, noise_params=None):
        self.qsvd = SimulatedQSVD(
            num_qubits=num_qubits,
            circuit_depth=circuit_depth,
            noise_params=noise_params
        )
        # Initialize with proper noise model
        self.qsvd.create_advanced_noise_model(noise_params)
        
        # Set up proper monitoring
        self.performance_monitor = PerformanceMonitor()
        self.gradient_monitor = GradientMonitor()
        
        # Initialize state dimensions
        self.state_size = 2**num_qubits + 1  # singular values + noise level
        self.action_size = self.qsvd.total_parameters
        
        # Initialize matrix dimensions
        self.matrix_dim = 2**num_qubits
        self.matrix = np.random.rand(self.matrix_dim, self.matrix_dim) + \
                     1j * np.random.rand(self.matrix_dim, self.matrix_dim)
        
        # Get true singular values for comparison
        self.true_singular_values = np.sort(np.abs(np.linalg.svd(self.matrix)[1]))[::-1]
        
        # Add running mean/std for normalization
        self.running_mean = RunningMeanStd(shape=self.state_size)
        self.running_reward = RunningMeanStd(shape=())
        
        # Initialize state
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        # Generate random initial matrix
        self.current_matrix = np.random.rand(2**self.qsvd.num_qubits, 2**self.qsvd.num_qubits) + \
                            1j * np.random.rand(2**self.qsvd.num_qubits, 2**self.qsvd.num_qubits)
        
        # Get initial singular values
        _, s, _ = np.linalg.svd(self.current_matrix)
        
        # Add noise level to state and normalize
        initial_state = np.concatenate([s, [self.qsvd.noise_level]])
        initial_state = self.running_mean.update(initial_state)
        
        return initial_state

class SimulatedQuantumEnv:
    '''
    Simulated quantum environment that mimics QSVD behavior without actual quantum circuits.
    '''
    def __init__(self, num_qubits, circuit_depth, noise_params=None):
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        
        # Calculate dimensions
        self.matrix_dim = 2**num_qubits
        
        # Define observation space shape (singular values + noise level)
        self.observation_shape = (self.matrix_dim + 1,)  # +1 for noise level
        
        # Add state_size attribute - it's the total dimension of the observation space
        self.state_size = self.matrix_dim + 1  # Same as observation_shape[0]
        
        # Add action_size attribute based on total parameters needed
        self.params_per_circuit = num_qubits * (1 + circuit_depth) * 3
        self.action_size = self.params_per_circuit * 2  # For both U and V circuits
        
        # Store noise_params as instance variable
        self.noise_params = noise_params or {
            't1': 45e-6,
            't2': 65e-6,
            'gate_times': {
                'single': 25e-9,
                'two': 45e-9,
            },
            'thermal_population': 0.015,
            'readout_error': 0.025,
            'crosstalk_strength': 0.035,
            'control_amplitude_error': 0.015,
            'control_frequency_drift': 0.006,
        }
        
        # Add noise_level initialization
        self.noise_level = self.noise_params.get('readout_error', 0.025)
        
        # Calculate total parameters needed
        self.params_per_circuit = num_qubits * (1 + circuit_depth) * 3
        self.total_params = self.params_per_circuit * 2  # For both U and V
        
        # Initialize matrix with correct dimensions
        self.matrix = np.random.rand(self.matrix_dim, self.matrix_dim) + \
                     1j * np.random.rand(self.matrix_dim, self.matrix_dim)
        
        # Initialize QSVD simulator
        self.qsvd_sim = SimulatedQSVD(num_qubits, circuit_depth, noise_params)
        
        # Get true singular values
        self.true_singular_values = np.sort(np.abs(np.linalg.svd(self.matrix)[1]))[::-1]
        
        # Initialize running statistics for normalization
        self.running_mean = RunningMeanStd(shape=self.observation_shape)
        self.running_reward = RunningMeanStd(shape=())
    
    def step(self, action):
        try:
            if len(action) != self.total_params:
                raise ValueError(f"Expected {self.total_params} parameters, got {len(action)}")
            
            # Ensure actions are within bounds
            action = np.clip(action, -np.pi, np.pi)
            
            # Split parameters for U and V circuits
            params_U = action[:self.params_per_circuit]
            params_V = action[self.params_per_circuit:]
            
            # Use proper circuit simulation
            noisy_values = self.qsvd_sim.simulate_svd(
                self.matrix,
                params_U.astype(np.float64),
                params_V.astype(np.float64)
            )
            
            # Ensure noisy_values has correct dimension
            noisy_values = noisy_values[:self.matrix_dim]
            
            # Calculate proper fidelity
            circuit_state = self.qsvd_sim._evaluate_circuit(self.qsvd_sim.circuit_U)
            fidelity = self.qsvd_sim._calculate_circuit_fidelity(circuit_state)
            
            # Calculate proper error
            error = np.mean(np.abs(noisy_values - self.true_singular_values))
            
            # Calculate reward using more nuanced function
            alpha = 0.5  # Weighting factor for fidelity
            reward = -error - alpha * (1 - fidelity)
            
            # State includes singular values and noise information
            state = np.concatenate([noisy_values, [self.qsvd_sim.noise_level]])
            
            # Normalize state and reward
            state = self.running_mean.normalize(state)
            reward = self.running_reward.normalize(reward)
            
            return state, reward, False, {
                'true_values': self.true_singular_values,
                'noisy_values': noisy_values,
                'fidelity': fidelity,
                'error': error
            }
            
        except Exception as e:
            print(f"Error in step: {str(e)}")
            # Fallback to safe values
            noisy_values = np.zeros(self.matrix_dim)
            state = np.concatenate([noisy_values, [self.qsvd_sim.noise_level]])
            return state, -10.0, True, {
                'true_values': self.true_singular_values,
                'noisy_values': noisy_values,
                'fidelity': 0.0,
                'error': 1.0
            }
    
    def _calculate_fidelity(self, noisy_values):
        """Calculate quantum state fidelity"""
        return np.abs(np.sum(np.sqrt(noisy_values * self.true_singular_values)))**2
    
    def reset(self):
        """Initialize environment state"""
        try:
            # Initialize matrix
            self.matrix = np.random.rand(self.matrix_dim, self.matrix_dim) + \
                         1j * np.random.rand(self.matrix_dim, self.matrix_dim)
            
            # Get true singular values
            self.true_singular_values = np.sort(np.abs(np.linalg.svd(self.matrix)[1]))[::-1]
            
            # Initial state: singular values + noise level
            initial_values = np.zeros(self.matrix_dim)  # Initialize with zeros
            state = np.concatenate([initial_values, [self.qsvd_sim.noise_level]])
            
            # Normalize state
            if hasattr(self, 'running_mean'):
                state = self.running_mean.normalize(state)
            
            return state
            
        except Exception as e:
            print(f"Error in reset: {e}")
            # Return safe fallback state
            return np.zeros(self.state_size)

class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOAgent, self).__init__()
        
        # 1. Use residual connections
        self.fc1 = nn.Linear(state_dim, 256)  # Wider layers
        self.ln1 = nn.LayerNorm(256)
        
        # 2. Add residual connection
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        
        self.fc3 = nn.Linear(256, action_dim)
        
        # 3. Better initialization
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        
        # 4. Adaptive log_std
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        # Add residual connections
        x = F.relu(self.ln1(self.fc1(state)))
        identity = x
        x = F.relu(self.ln2(self.fc2(x)))
        x = x + identity  # Residual connection
        
        mean = torch.tanh(self.fc3(x)) * np.pi
        std = torch.clamp(self.log_std.exp(), min=1e-6, max=1.0)
        
        return mean, std
    
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            mean, std = self.forward(state)
            
            # Safety checks
            if torch.isnan(mean).any() or torch.isnan(std).any():
                print("Warning: NaN detected before action selection")
                mean = torch.zeros_like(mean)
                std = torch.ones_like(std)
            
            try:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                return action.detach().numpy(), dist.log_prob(action).sum().item()
            except ValueError as e:
                print(f"Error in action selection: {e}")
                # Return safe fallback values
                return np.zeros(self.action_dim), 0.0

def train_agent(episodes=1000, num_qubits=4, circuit_depth=4, noise_params=None):
    """Training loop with proper state handling"""
    env = SimulatedQuantumEnv(num_qubits, circuit_depth, noise_params)
    agent = QSVDAgent(env.state_size, env.action_size)
    
    metrics = initialize_metrics()
    
    for episode in range(episodes):
        state = env.reset()  # Ensure this returns a valid state
        if state is None:
            print(f"Error: Episode {episode} - reset() returned None state")
            continue
            
        episode_reward = 0
        done = False
        
        while not done:
            try:
                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                
                if next_state is None:
                    print(f"Error: Episode {episode} - step() returned None state")
                    break
                    
                # Store experience and update agent
                agent.store_experience(state, action, 0.0, reward, next_state, done, 0.0)
                state = next_state
                episode_reward += reward
                
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                break
        
        # Update agent and metrics
        update_metrics(metrics, episode_reward, info)
        
    return agent, metrics

def plot_training_metrics(metrics, save_path):
    """
    Create detailed visualization of training metrics
    """
    plt.figure(figsize=(20, 10))
    
    # Rewards
    plt.subplot(2, 3, 1)
    plt.plot(metrics['rewards'])
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Errors
    plt.subplot(2, 3, 2)
    plt.plot(metrics['errors'])
    plt.title('SVD Error Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Mean Absolute Error')
    
    # Fidelity
    plt.subplot(2, 3, 3)
    plt.plot(metrics['fidelity_history'])
    plt.title('Quantum Fidelity')
    plt.xlabel('Episode')
    plt.ylabel('Fidelity')
    
    # Circuit Quality
    plt.subplot(2, 3, 4)
    depth_efficiency = [m['depth_efficiency'] for m in metrics['circuit_quality']]
    plt.plot(depth_efficiency)
    plt.title('Circuit Depth Efficiency')
    plt.xlabel('Episode')
    plt.ylabel('Efficiency')
    
    # Noise Levels
    plt.subplot(2, 3, 5)
    plt.plot(metrics['noise_levels'])
    plt.title('Noise Level Evolution')
    plt.xlabel('Episode')
    plt.ylabel('Noise Level')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.std = np.ones(shape)
        self.count = 1e-4
        
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_std = np.std(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / (self.count + batch_count)
        self.std = np.sqrt(self.std**2 + batch_std**2 + delta**2 * 
                          self.count * batch_count / (self.count + batch_count))
        self.count += batch_count
        
    def normalize(self, x):
        return (x - self.mean) / (self.std + 1e-8)

