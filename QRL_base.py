# =====================================================================
# Quantum Reinforcement Learning with PPO for QSVD
# =====================================================================
# Author: Boden Moraski
# Description: Main script integrating the QSVD functions with a PPO-based
#              reinforcement learning agent to optimize quantum circuits
#              for singular value decomposition in noisy environments.
# =====================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this line
import torch.optim as optim
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from qiskit_aer import AerSimulator
from qsvdfuncs import (
    create_parameterized_circuit, 
    get_unitary, 
    loss_function, 
    objective, 
    optimize_vqsvd, 
    plot_results, 
    compare_with_classical_svd,
    encode_matrix_as_state,
    get_gradient,
    analyze_circuit_expressiveness,
    run_vqsvd
)
from refQSVD import VQSVD
from qiskit.quantum_info.operators import Kraus

# Define the Quantum Environment Using Qiskit
class QuantumEnv:
    """
    Quantum environment that simulates noisy quantum circuits.
    The environment interacts with the RL agent, providing feedback
    in the form of quantum fidelity.
    """
    def __init__(self, num_qubits, circuit_depth):
        self.backend = AerSimulator()
        self.noise_model = self._create_noise_model()
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.rank = 2**num_qubits  # Full rank for simplicity, can be adjusted
        self.circuit_U = create_parameterized_circuit(num_qubits, circuit_depth, 'U')
        self.circuit_V = create_parameterized_circuit(num_qubits, circuit_depth, 'V')
        self.matrix = np.random.rand(2**num_qubits, 2**num_qubits) + \
                      1j * np.random.rand(2**num_qubits, 2**num_qubits)

    def _create_noise_model(self):
        """
        Creates a noise model to simulate realistic quantum gate errors.
        """
        noise_model = NoiseModel()
        
        # Single-qubit gate error
        error_1 = thermal_relaxation_error(t1=50, t2=30, time=10)
        
        # Two-qubit gate error (e.g., for CX gates)
        error_2 = error_1.expand(error_1)
        
        # Add errors to the noise model
        noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])  # Single-qubit gates
        noise_model.add_all_qubit_quantum_error(error_2, ['cx'])  # Two-qubit gates
        
        return noise_model

    def step(self, action):
        """
        Applies the action (circuit parameters) to the environment,
        computes the reward based on QSVD performance, and returns
        the next state, reward, and done flag.
        """
        # Split action into parameters for U and V circuits
        params_U = action[:len(self.circuit_U.parameters)]
        params_V = action[len(self.circuit_U.parameters):]
        
        # Calculate the loss as negative reward
        loss = loss_function(
            self.matrix, 
            self.circuit_U, 
            self.circuit_V, 
            params_U, 
            params_V, 
            rank=self.rank, 
            noise_model=self.noise_model
        )
        reward = -loss  # Minimize loss
        
        # Ensure the state is a valid float
        state = np.array([max(min(loss, 1e6), -1e6)])  # Clip to avoid extreme values
        
        # Assuming episode ends after each step for simplicity
        done = True
        
        # Optionally, provide additional info
        info = {}
        
        return state, reward, done, info

    def reset(self):
        """
        Resets the environment to an initial state.
        """
        # Initialize with random parameters
        params_U = np.random.uniform(-np.pi, np.pi, len(self.circuit_U.parameters))
        params_V = np.random.uniform(-np.pi, np.pi, len(self.circuit_V.parameters))
        action = np.concatenate([params_U, params_V])
        state, reward, done, info = self.step(action)
        return state

# =====================================================================
# 3. Define the PPO Reinforcement Learning Agent
# =====================================================================
class PPOAgent(nn.Module):
    """
    Proximal Policy Optimization (PPO) agent to optimize quantum gate
    operations in noisy environments.
    """

    def __init__(self, state_dim, action_dim):
        super(PPOAgent, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        # self.ln1 = nn.LayerNorm(64)  # Removed LayerNorm
        self.fc2 = nn.Linear(64, 64)
        # self.ln2 = nn.LayerNorm(64)  # Removed LayerNorm
        self.mean_layer = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.fc1(x)
        print("After fc1:", x)
        # x = self.ln1(x)
        # print("After ln1:", x)
        x = F.relu(x)
        print("After relu1:", x)
        x = self.fc2(x)
        print("After fc2:", x)
        # x = self.ln2(x)
        # print("After ln2:", x)
        x = F.relu(x)
        print("After relu2:", x)
        mean = self.mean_layer(x)
        print("Final mean:", mean)
        return mean

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean = self(state)
        std = torch.exp(self.log_std).clamp(min=1e-6, max=1)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -np.pi, np.pi)
        log_prob = dist.log_prob(action).sum()
        return action.detach().numpy().flatten(), log_prob.item()

# =====================================================================
# 5. Main Training Loop: Agent-Environment Interaction
# =====================================================================
def train_agent(episodes=100, num_qubits=2, circuit_depth=5, batch_size=10):
    """
    Trains the PPO agent to adaptively control quantum gates.
    Collects experiences over a batch of episodes before performing updates.
    """
    env = QuantumEnv(num_qubits, circuit_depth)
    state_dim = 1  # We're using a 1D state now
    action_dim = len(env.circuit_U.parameters) + len(env.circuit_V.parameters)
    
    agent = PPOAgent(state_dim, action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=0.0001)  # Reduced learning rate

    # Hyperparameters for PPO
    gamma = 0.99
    epsilon = 0.2
    tau = 0.95
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5

    # Storage for PPO
    memory = []  # This should store (state, action, log_prob, reward, next_state, done)
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Scale reward
            scaled_reward = np.clip(reward, -1, 1)
            
            memory.append((state, action, log_prob, scaled_reward, next_state, done))
            state = next_state
            episode_reward += reward

        print(f"Episode {episode + 1}: Reward = {episode_reward}")

        # Perform PPO update every 'batch_size' episodes
        if (episode + 1) % batch_size == 0 and len(memory) > 0:
            states, actions, old_log_probs, rewards, _, _ = zip(*memory)
            
            # Convert lists to numpy arrays first for efficiency
            states = np.array(states)
            actions = np.array(actions)
            old_log_probs = np.array(old_log_probs)
            rewards = np.array(rewards)

            # Convert to tensors
            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            old_log_probs = torch.FloatTensor(old_log_probs)
            rewards = torch.FloatTensor(rewards)

            # Compute returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            if torch.isnan(returns).any():
                print("Warning: NaN detected in returns. Skipping this batch update.")
                memory = []
                continue
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Detect anomalies
            with torch.autograd.detect_anomaly():
                # Multiple update epochs
                for _ in range(10):
                    mean = agent(states)
                    std = torch.exp(agent.log_std).clamp(min=1e-6, max=1)
                    dist = torch.distributions.Normal(mean, std)
                    new_log_probs = dist.log_prob(actions).sum(-1)

                    # Compute ratio and surrogate loss
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * returns
                    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * returns
                    loss = -torch.min(surr1, surr2).mean()

                    print(f"Current Loss: {loss.item()}")

                    if torch.isnan(loss):
                        print("Loss is NaN. Skipping this batch update.")
                        break

                    # Update network
                    optimizer.zero_grad()
                    loss.backward()

                    # Check for NaNs in gradients
                    for name, param in agent.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                print(f"Warning: NaN detected in gradients of {name}. Skipping this batch update.")
                                break

                    # Gradient clipping
                    nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)

                    optimizer.step()

                    # Check for NaNs in parameters after update
                    for name, param in agent.named_parameters():
                        if torch.isnan(param).any():
                            print(f"Error: NaN detected in weights of {name} after update. Stopping training.")
                            return

            # Clear memory after update
            memory = []

# =====================================================================
# 6. Entry Point
# =====================================================================
if __name__ == "__main__":
    """
    Entry point for the program. This is where the training process begins.
    """
    print("Starting training...")
    train_agent(episodes=100, num_qubits=2, circuit_depth=5, batch_size=10)  # Increased from 3 to 5
    print("Training complete.")
    
    # Create an instance of QuantumEnv
    env = QuantumEnv(num_qubits=2, circuit_depth=5)
    
    # Example usage of run_vqsvd after training
    NUM_QUBITS = 2
    CIRCUIT_DEPTH = 5
    RANK = 2**NUM_QUBITS
    LR = 0.001
    MAX_ITERS = 100

    # Generate a random matrix
    M = np.random.rand(2**NUM_QUBITS, 2**NUM_QUBITS) + 1j * np.random.rand(2**NUM_QUBITS, 2**NUM_QUBITS)

    # Perform QSVD using run_vqsvd
    comparison = run_vqsvd(
        matrix=M, 
        rank=RANK, 
        num_qubits=NUM_QUBITS, 
        circuit_depth=CIRCUIT_DEPTH, 
        lr=LR, 
        max_iters=MAX_ITERS,
        noise_model=env.noise_model  # Now env is defined
    )

    print("QSVD Comparison Results:")
    for key, value in comparison.items():
        print(f"{key}: {value}")
