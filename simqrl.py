import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.linalg import svd as classical_svd
from simqsvd import SimulatedQSVD, PerformanceMonitor

class SimulatedQuantumEnv:
    """
    Simulated quantum environment that mimics QSVD behavior without actual quantum circuits.
    """
    def __init__(self, num_qubits, circuit_depth, noise_params=None):
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        
        # Default noise parameters if none provided
        self.noise_params = noise_params or {
            't1': 50e-6,                # T1 relaxation time
            't2': 70e-6,               # T2 dephasing time
            'gate_times': {            # Gate operation times
                'single': 20e-9,       # Single-qubit gate time
                'two': 40e-9,          # Two-qubit gate time
            },
            'thermal_population': 0.01, # Thermal excitation probability
            'readout_error': 0.02,     # Measurement readout error
            'crosstalk_strength': 0.03, # Inter-qubit crosstalk
            'control_amplitude_error': 0.01,  # Gate amplitude error
            'control_frequency_drift': 0.005, # Frequency drift
        }
        
        # Calculate overall noise level from parameters
        self.noise_level = np.mean([
            self.noise_params['thermal_population'],
            self.noise_params['readout_error'],
            self.noise_params['crosstalk_strength'],
            self.noise_params['control_amplitude_error'],
            self.noise_params['control_frequency_drift']
        ])
        
        # Initialize QSVD simulator with noise parameters
        self.qsvd_sim = SimulatedQSVD(num_qubits, circuit_depth, self.noise_params)
        
        # Calculate number of parameters per circuit
        self.params_per_circuit = num_qubits * circuit_depth
        
        # Generate random matrix
        self.matrix = np.random.rand(2**num_qubits, 2**num_qubits) + \
                     1j * np.random.rand(2**num_qubits, 2**num_qubits)
        
        # Get true singular values
        self.true_singular_values = self.qsvd_sim.get_true_singular_values(self.matrix)
    
    def step(self, action):
        # Split action into U and V parameters
        params_U = action[:self.params_per_circuit]
        params_V = action[self.params_per_circuit:]
        
        # Get noisy singular values using quantum simulation
        noisy_values = self.qsvd_sim.simulate_svd(
            self.matrix, params_U, params_V
        )
        
        # Calculate reward based on accuracy
        reward = -np.mean((noisy_values - self.true_singular_values)**2)
        
        # State includes singular values and noise information
        state = np.concatenate([noisy_values, [self.noise_level]])
        
        info = {
            'true_values': self.true_singular_values,
            'noisy_values': noisy_values,
            'quantum_fidelity': self._calculate_fidelity(noisy_values)
        }
        
        return state, reward, False, info
    
    def _calculate_fidelity(self, noisy_values):
        """Calculate quantum state fidelity"""
        return np.abs(np.sum(np.sqrt(noisy_values * self.true_singular_values)))**2
    
    def reset(self):
        """Reset the environment"""
        # Get initial singular values
        initial_values = np.sort(np.abs(np.diagonal(self.matrix)))[::-1]
        
        # Apply noise using the QSVD simulator
        noisy_initial = self.qsvd_sim.simulate_svd(
            self.matrix,
            np.random.uniform(-np.pi, np.pi, self.params_per_circuit),  # Random initial params for U
            np.random.uniform(-np.pi, np.pi, self.params_per_circuit)   # Random initial params for V
        ) # TODO: give a better implementation later
        
        # Return state vector (noisy singular values + noise level)
        return np.concatenate([noisy_initial, [self.noise_level]])

class PPOAgent(nn.Module):
    """
    PPO agent optimized for noise reduction in simulated QSVD
    """
    def __init__(self, state_dim, action_dim):
        super(PPOAgent, self).__init__()
        # Larger network for better noise handling
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc3(x)
        std = self.log_std.exp()
        return mean, std
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action.detach().numpy(), dist.log_prob(action).sum().item()

def train_agent(episodes=1000, num_qubits=4, circuit_depth=4, noise_params=None):
    """
    Train the PPO agent with enhanced monitoring and debugging
    """
    # Create environment and monitoring tools
    env = SimulatedQuantumEnv(num_qubits, circuit_depth, noise_params)
    performance_monitor = PerformanceMonitor()
    
    # Initialize tracking metrics
    state_dim = 2**num_qubits + 1
    action_dim = num_qubits * circuit_depth * 2
    
    agent = PPOAgent(state_dim, action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    
    # Enhanced history tracking
    training_metrics = {
        'rewards': [],
        'errors': [],
        'noise_levels': [],
        'circuit_quality': [],
        'gradient_stats': [],
        'fidelity_history': []
    }
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        # Collect experience with enhanced monitoring
        action, log_prob = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        # Monitor circuit quality
        circuit_metrics = env.qsvd_sim.analyze_circuit_quality(
            env.qsvd_sim.circuit_U,
            env.qsvd_sim.circuit_V,
            action
        )
        
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
        training_metrics['circuit_quality'].append(circuit_metrics)
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

if __name__ == "__main__":
    # Define custom noise parameters (optional)
    noise_params = {
        't1': 45e-6,                # Slightly worse T1
        't2': 65e-6,               # Slightly worse T2
        'gate_times': {
            'single': 25e-9,       # Slower single-qubit gates
            'two': 45e-9,          # Slower two-qubit gates
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
    
    # Generate final report
    print("\nFinal Training Report")
    print("=" * 50)
    print(f"Final Average Reward: {np.mean(training_metrics['rewards'][-100:]):.4f}")
    print(f"Final Average Error: {np.mean(training_metrics['errors'][-100:]):.4f}")
    print(f"Final Average Fidelity: {np.mean(training_metrics['fidelity_history'][-100:]):.4f}")
    print(f"Training Stability: {np.std(training_metrics['rewards'][-100:]):.4f}")
    print("=" * 50)
