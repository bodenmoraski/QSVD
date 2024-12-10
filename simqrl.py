import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.linalg import svd as classical_svd
from simqsvd import SimulatedQSVD

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
        
        # Calculate correct number of parameters per circuit
        # 3 gates (rx, ry, rz) per qubit per layer, including initial layer
        self.params_per_circuit = self.num_qubits * (1 + self.circuit_depth) * 3
        
        # Initialize QSVD simulator with noise parameters
        self.qsvd_sim = SimulatedQSVD(num_qubits, circuit_depth)
        
        # Generate random matrix
        self.matrix = np.random.rand(2**num_qubits, 2**num_qubits) + \
                     1j * np.random.rand(2**num_qubits, 2**num_qubits)
        
        # Get true singular values
        self.true_singular_values = self.qsvd_sim.get_true_singular_values(self.matrix)
    
    def step(self, action):
        # Adjust action splitting based on correct parameter count
        params_U = action[:self.params_per_circuit]
        params_V = action[self.params_per_circuit:]
        
        # Get noisy singular values using quantum simulation
        try:
            noisy_values = self.qsvd_sim.simulate_svd(
                self.matrix, params_U, params_V
            )
        except ValueError as e:
            print(f"Step failed: {str(e)}")
            # Fallback to noisy diagonal values
            noisy_values = np.sort(np.abs(np.diagonal(self.matrix)))[::-1]
            noisy_values += np.random.normal(0, self.noise_level, size=len(noisy_values))
        
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
    Train the PPO agent to reduce noise in simulated QSVD
    """
    # Create environment
    env = SimulatedQuantumEnv(num_qubits, circuit_depth, noise_params)
    
    # Calculate correct dimensions
    state_dim = 2**num_qubits + 1  # singular values + noise level
    action_dim = env.params_per_circuit * 2  # Parameters for both U and V circuits
    
    # Initialize agent with correct dimensions
    agent = PPOAgent(state_dim, action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    
    rewards_history = []
    error_history = []
    noise_history = []  # Track noise levels over time
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        # Collect experience
        action, log_prob = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        # Convert reward to PyTorch tensor
        loss = -torch.tensor(reward, requires_grad=True)
        
        # Update agent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record history (using numpy for storage)
        rewards_history.append(float(reward))
        error = np.mean(np.abs(info['noisy_values'] - info['true_values']))
        error_history.append(error)
        noise_history.append(info.get('noise_level', 0))
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:]) if episode >= 100 else np.mean(rewards_history)
            avg_error = np.mean(error_history[-100:]) if episode >= 100 else np.mean(error_history)
            print(f"Episode {episode}")
            print(f"Avg Reward: {avg_reward:.4f}")
            print(f"Avg Error: {avg_error:.4f}")
            print("Current Noise Parameters:")
            for key, value in env.noise_params.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
            print("-" * 50)
    
    return agent, rewards_history, error_history, noise_history

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
    
    # Train agent with advanced noise model
    agent, rewards, errors, noise_levels = train_agent(
        episodes=1000,
        num_qubits=4,
        circuit_depth=4,
        noise_params=noise_params
    )
    
    # Plot results with noise information
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(errors)
    plt.title('SVD Error Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Mean Absolute Error')
    
    plt.subplot(1, 3, 3)
    plt.plot(noise_levels)
    plt.title('Noise Level Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Noise Level')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()
