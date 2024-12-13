import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.linalg import svd as classical_svd
from simqsvd import SimulatedQSVD, PerformanceMonitor
from running_mean_std import RunningMeanStd

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
            
            # Get noisy singular values
            noisy_values = self.qsvd_sim.simulate_svd(
                self.matrix,
                params_U.astype(np.float64),
                params_V.astype(np.float64)
            )
            
            # Ensure noisy_values has correct dimension
            noisy_values = noisy_values[:self.matrix_dim]
            
            # Calculate reward
            reward = -np.mean((noisy_values - self.true_singular_values)**2)
            
            # State includes singular values and noise information
            state = np.concatenate([noisy_values, [self.qsvd_sim.noise_level]])
            
            # Normalize state and reward
            state = self.running_mean.normalize(state)
            reward = self.running_reward.normalize(reward)
            
            return state, reward, False, {
                'true_values': self.true_singular_values,
                'noisy_values': noisy_values,
                'quantum_fidelity': self._calculate_fidelity(noisy_values)
            }
            
        except Exception as e:
            print(f"Error in step: {str(e)}")
            # Fallback to safe values
            noisy_values = np.zeros(self.matrix_dim)
            state = np.concatenate([noisy_values, [self.qsvd_sim.noise_level]])
            return state, -1.0, False, {
                'true_values': self.true_singular_values,
                'noisy_values': noisy_values,
                'quantum_fidelity': 0.0
            }
    
    def _calculate_fidelity(self, noisy_values):
        """Calculate quantum state fidelity"""
        return np.abs(np.sum(np.sqrt(noisy_values * self.true_singular_values)))**2
    
    def reset(self):
        """Reset the environment"""
        # Get initial singular values
        initial_values = np.sort(np.abs(np.diagonal(self.matrix)))[::-1]
        
        try:
            # Apply noise using the QSVD simulator with correct parameter count
            noisy_initial = self.qsvd_sim.simulate_svd(
                self.matrix,
                np.random.uniform(-np.pi, np.pi, self.params_per_circuit),
                np.random.uniform(-np.pi, np.pi, self.params_per_circuit)
            )
        except Exception as e:
            print(f"Reset failed: {str(e)}")
            # Fallback to initial values with synthetic noise
            noisy_initial = initial_values + np.random.normal(
                0, self.noise_level, size=len(initial_values)
            )
        
        return np.concatenate([noisy_initial, [self.noise_level]])

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
    """
    Train the PPO agent with enhanced monitoring and debugging
    """
    # Create environment and monitoring tools
    env = SimulatedQuantumEnv(num_qubits, circuit_depth, noise_params)
    performance_monitor = PerformanceMonitor()
    
    # Calculate correct dimensions
    state_dim = 2**num_qubits + 1  # +1 for noise level
    params_per_circuit = num_qubits * (1 + circuit_depth) * 3
    action_dim = params_per_circuit * 2  # Multiply by 2 for both U and V circuits
    
    # Initialize agent with correct dimensions
    agent = PPOAgent(state_dim, action_dim)
    optimizer = optim.AdamW(
        agent.parameters(),
        lr=3e-4,
        weight_decay=0.01,  # L2 regularization
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5,
        patience=50,
        verbose=True
    )
    max_grad_norm = 0.5
    
    # Enhanced history tracking
    training_metrics = {
        'rewards': [],
        'errors': [],
        'noise_levels': [],
        'circuit_quality': [],
        'gradient_stats': [],
        'fidelity_history': []
    }
    
    # Initialize circuit metrics
    circuit_metrics = {
        'gradient_vanishing': False,
        'gradient_exploding': False,
        'depth_efficiency': 1.0,
        'parameter_norm': 0.0,
        'gradient_norm': 0.0
    }
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        # Collect experience with enhanced monitoring
        action, log_prob = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        # Update circuit metrics
        with torch.no_grad():
            # Calculate parameter norms
            param_norm = sum(p.norm().item() for p in agent.parameters())
            grad_norm = sum(p.grad.norm().item() if p.grad is not None else 0 
                          for p in agent.parameters())
            
            circuit_metrics.update({
                'gradient_vanishing': grad_norm < 1e-7,
                'gradient_exploding': grad_norm > 1e3,
                'depth_efficiency': 1.0 / circuit_depth,  # Simple metric, could be more sophisticated
                'parameter_norm': param_norm,
                'gradient_norm': grad_norm
            })
        
        # Only print detailed info every 50 episodes
        if episode % 50 == 0:
            print(f"\nEpisode {episode} Details:")
            print("=" * 50)
            
            # Circuit Parameters
            print("\nCircuit Configuration:")
            print(f"  Number of qubits: {num_qubits}")
            print(f"  Circuit depth: {circuit_depth}")
            
            # Singular Values Comparison
            true_values = info.get('true_values', [])
            noisy_values = info.get('noisy_values', [])
            print("\nSingular Values:")
            print("  True      Noisy     Error(%)")
            print("-" * 35)
            for true, noisy in zip(true_values, noisy_values):
                error_pct = abs(true - noisy) / true * 100 if true != 0 else 0
                print(f"  {true:8.4f} {noisy:8.4f} {error_pct:8.2f}%")
            
            # Performance Metrics
            current_error = np.mean(np.abs(np.array(true_values) - np.array(noisy_values)))
            print("\nPerformance Metrics:")
            print(f"  Current Error: {current_error:.4f}")
            print(f"  Recent Reward: {reward:.4f}")
            print(f"  Fidelity: {info.get('quantum_fidelity', 0):.4f}")
            
            # Noise Analysis
            if noise_params:
                print("\nNoise Parameters:")
                for param, value in noise_params.items():
                    print(f"  {param}: {value}")
            
            print("=" * 50)
        
        # Update performance monitor
        performance_monitor.update(
            info['noisy_values'],
            info['true_values'],
            next_state,
            env.noise_level
        )
        
        # Record detailed metrics
        training_metrics['rewards'].append(float(reward))
        training_metrics['errors'].append(np.mean(np.abs(info['noisy_values'] - info['true_values'])))
        training_metrics['noise_levels'].append(env.noise_level)
        training_metrics['fidelity_history'].append(info['quantum_fidelity'])
        
        # Detailed logging every N episodes
        if episode % 50 == 0 and episode > 0:
            perf_report = performance_monitor.generate_report()
            print(f"\nEpisode {episode} Detailed Report:")
            print("=" * 50)
            print(f"Performance Metrics:")
            print(f"  Average Error: {perf_report['avg_error']:.4f}")
            print(f"  Fidelity Trend: {'Improving' if perf_report['fidelity_trend'] > 0 else 'Degrading'}")
            print(f"  Noise-Error Correlation: {perf_report['noise_correlation']:.4f}")
            
            print("\nCircuit Quality Metrics:")
            print(f"  Gradient Vanishing: {circuit_metrics['gradient_vanishing']}")
            print(f"  Gradient Exploding: {circuit_metrics['gradient_exploding']}")
            print(f"  Depth Efficiency: {circuit_metrics['depth_efficiency']:.4f}")
            
            print("\nNoise Analysis:")
            for key, value in env.noise_params.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
            
            print("\nTraining Statistics:")
            print(f"  Recent Reward Avg: {np.mean(training_metrics['rewards'][-50:]):.4f}")
            print(f"  Recent Error Avg: {np.mean(training_metrics['errors'][-50:]):.4f}")
            print(f"  Recent Fidelity Avg: {np.mean(training_metrics['fidelity_history'][-50:]):.4f}")
            print("=" * 50)
        
        # Regular progress update
        if episode % 100 == 0:
            avg_reward = np.mean(training_metrics['rewards'][-100:]) if episode >= 100 else np.mean(training_metrics['rewards'])
            avg_error = np.mean(training_metrics['errors'][-100:]) if episode >= 100 else np.mean(training_metrics['errors'])
            print(f"\nEpisode {episode} Summary:")
            print(f"Avg Reward: {avg_reward:.4f}")
            print(f"Avg Error: {avg_error:.4f}")
            print("-" * 50)
        
        # Implement gradient clipping
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        
        # Track gradients for monitoring
        grad_norm = torch.nn.utils.clip_grad_norm_(
            agent.parameters(), 
            float('inf')
        ).item()
        
        # Update circuit metrics with gradient info
        circuit_metrics.update({
            'gradient_vanishing': grad_norm < 1e-7,
            'gradient_exploding': grad_norm > 1e3,
            'gradient_norm': grad_norm
        })
        
        # Update learning rate based on performance
        scheduler.step(reward)
    
    # Generate final plots
    plot_training_metrics(training_metrics, save_path='detailed_training_results.png')
    
    return agent, training_metrics

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

if __name__ == "__main__":
    try:
        # Define custom noise parameters (optional)
        noise_params = {
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
        
        # Train agent with enhanced monitoring
        agent, training_metrics = train_agent(
            episodes=1000,
            num_qubits=4,
            circuit_depth=4,
            noise_params=noise_params
        )
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise
    
    # Generate final report
    print("\nFinal Training Report")
    print("=" * 50)
    print(f"Final Average Reward: {np.mean(training_metrics['rewards'][-100:]):.4f}")
    print(f"Final Average Error: {np.mean(training_metrics['errors'][-100:]):.4f}")
    print(f"Final Average Fidelity: {np.mean(training_metrics['fidelity_history'][-100:]):.4f}")
    print(f"Training Stability: {np.std(training_metrics['rewards'][-100:]):.4f}")
    print("=" * 50)
