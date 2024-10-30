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
import matplotlib.pyplot as plt
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
from qiskit.exceptions import QiskitError

# Define the Quantum Environment Using Qiskit
class QuantumEnv:
    """
    Quantum environment that simulates noisy quantum circuits.
    The environment interacts with the RL agent, providing feedback
    in the form of quantum fidelity.
    """
    def __init__(self, num_qubits, circuit_depth):
        self.backend = AerSimulator(method='statevector')  # Use statevector simulator
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
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc3(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std  # This is returning a tuple

    def get_action_distribution(self, state):
        mean, std = self.forward(state)
        return torch.distributions.Normal(mean, std)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        dist = self.get_action_distribution(state)
        action = dist.sample()
        action = torch.clamp(action, -np.pi, np.pi)
        log_prob = dist.log_prob(action).sum(-1)
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

    # Storage for PPO and visualization
    memory = []
    episode_rewards = []
    average_losses = []
    best_reward = float('-inf')
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_losses = []
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Scale reward
            scaled_reward = np.clip(reward, -1, 1)
            
            memory.append((state, action, log_prob, scaled_reward, next_state, done))
            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)
        
        # Perform PPO update every 'batch_size' episodes
        if (episode + 1) % batch_size == 0 and len(memory) > 0:
            states, actions, old_log_probs, rewards, next_states, dones = zip(*memory)
            
            # Convert to tensors
            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            old_log_probs = torch.FloatTensor(old_log_probs)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            # Compute returns
            returns = []
            R = 0
            for r, d in zip(reversed(rewards), reversed(dones)):
                R = r + gamma * R * (1 - d)
                returns.insert(0, R)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Multiple update epochs
            for _ in range(10):
                # Get current log probs and values
                mean, std = agent(states)
                print("Type of mean:", type(mean))
                print("Shape of mean:", mean.shape if hasattr(mean, 'shape') else "No shape")
                print("Type of std:", type(std))
                print("Shape of std:", std.shape if hasattr(std, 'shape') else "No shape")
                dist = torch.distributions.Normal(mean, std)
                current_log_probs = dist.log_prob(actions).sum(-1)

                # Compute ratio and surrogate loss
                ratio = torch.exp(current_log_probs - old_log_probs)
                surr1 = ratio * returns
                surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * returns
                loss = -torch.min(surr1, surr2).mean()

                # Update network
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

                episode_losses.append(loss.item())

            # Clear memory after update
            memory = []

        # Compute average loss for this episode
        avg_loss = np.mean(episode_losses) if episode_losses else None
        if avg_loss is not None:
            average_losses.append(avg_loss)

        print(f"Episode {episode + 1}: Reward = {episode_reward}, Avg Loss = {avg_loss}")

        # Check for early stopping conditions
        if len(episode_rewards) >= 10:
            avg_reward = np.mean(episode_rewards[-10:])
            if avg_reward > 0.95:
                print(f"Early stopping: Average reward {avg_reward} > 0.95")
                break

        if avg_loss is not None and avg_loss < 1e-4:
            print(f"Early stopping: Average loss {avg_loss} < 1e-4")
            break

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.state_dict(), 'best_model.pth')

    return agent, episode_rewards, average_losses

def plot_training_progress(episode_rewards, average_losses):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(average_losses)
    plt.title('Average Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

def plot_final_circuit_analysis(env, agent):
    # Generate a random input state
    input_state = np.random.rand(2**env.num_qubits) + 1j * np.random.rand(2**env.num_qubits)
    input_state /= np.linalg.norm(input_state)

    # Apply the trained circuit to the input state
    circuit = env.circuit_U.copy()
    circuit.compose(env.circuit_V, inplace=True)
    
    print(f"Number of circuit parameters: {len(circuit.parameters)}")
    
    # Get the final parameters
    params = agent.state_dict()
    final_params = np.concatenate([params[key].numpy().flatten() for key in params if 'weight' in key or 'bias' in key])
    print(f"Number of final parameters: {len(final_params)}")
    
    # Create a dictionary mapping parameters to values
    param_dict = dict(zip(circuit.parameters, final_params[:len(circuit.parameters)]))
    
    # Bind the parameters to the circuit
    bound_circuit = circuit.assign_parameters(param_dict)
    
    # Initialize the input state
    bound_circuit.initialize(input_state)
    
    # Configure the backend correctly
    backend = AerSimulator(
        method='statevector',
        shots=1000,
        memory=True
    )
    
    # Add measurement basis
    meas_circuit = bound_circuit.copy()
    meas_circuit.save_statevector()  # This is the correct way to save the statevector
    
    # Run the simulation
    job = backend.run(meas_circuit)
    result = job.result()
    
    try:
        # Try to get statevector
        output_state = result.get_statevector()
    except QiskitError:
        print("Unable to get statevector. Using unitary evolution instead.")
        # If statevector fails, simulate the circuit evolution manually
        unitary = Operator(bound_circuit).data
        output_state = unitary @ input_state
    
    # Plot the results
    real_output = np.real(output_state)
    imag_output = np.imag(output_state)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(real_output)), real_output, alpha=0.5, label='Real', linewidth=1.0)
    plt.bar(range(len(imag_output)), imag_output, alpha=0.5, label='Imaginary', linewidth=1.0)
    plt.xlabel("Basis State")
    plt.ylabel("Amplitude")
    plt.title("Final Circuit Output State")
    plt.legend()
    plt.tight_layout()
    plt.savefig('final_circuit_analysis.png')
    plt.close()

# =====================================================================
# 6. Entry Point
# =====================================================================
if __name__ == "__main__":
    """
    Entry point for the program. This is where the training process begins.
    """
    print("Starting training...")
    agent, episode_rewards, average_losses = train_agent(episodes=100, num_qubits=2, circuit_depth=5, batch_size=10)
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
        noise_model=env.noise_model
    )

    print("QSVD Comparison Results:")
    for key, value in comparison.items():
        if isinstance(value, (int, float, np.ndarray)):
            print(f"{key}: {value}")

    # Additional visualization
    plt.figure(figsize=(10, 6))
    plt.plot(comparison['Loss History'])
    plt.title('QSVD Loss History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig('qsvd_loss_history.png')
    plt.close()

    # Singular value comparison
    plt.figure(figsize=(10, 6))
    plt.plot(comparison['Singular Values History'][-1], label='VQSVD')
    plt.plot(comparison['Classical Singular Values'], label='Classical SVD')
    plt.title('Singular Value Comparison')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.legend()
    plt.savefig('singular_value_comparison.png')
    plt.close()

    # Plot training progress
    plot_training_progress(episode_rewards, average_losses)

    # Analyze final circuit
    plot_final_circuit_analysis(env, agent)

