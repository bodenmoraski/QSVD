# =====================================================================
# Adaptive Quantum Control with Reinforcement Learning (PPO) & QSVD
# =====================================================================
# Author: Boden Moraski
# Description: Boilerplate code for the quantum control project.
# The purpose of this code is to demonstrate the core structure
# and design of the project, including PPO-based RL, quantum circuits,
# and QSVD-based optimization for noisy quantum operations.
# =====================================================================
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from qiskit.compiler import transpiler, transpile
import torch
import torch.nn as nn
import torch.optim as optim
#import quantumsvd as qsvd
# import refQSVD as r_qsvd
from qsvdfuncs import create_parameterized_circuit, get_unitary, run_vqsvd, analyze_circuit_expressiveness, check_unitarity

# Define a placeholder for the QSVD module
# (Actual QSVD logic will go here in later iterations)
def qsvd_decompose(operator):
    """
    Decomposes the given quantum operator into unitary matrices
    using QSVD. This function will serve as a placeholder for now.
    Returns: (Q, Sigma, V_dagger) - components of the QSVD.
    """
    Q = np.identity(operator.shape[0])
    Sigma = np.diag(np.ones(operator.shape[0]))
    V_dagger = Q.T
    return Q, Sigma, V_dagger

# =====================================================================
# 2. Define the Quantum Environment Using Qiskit
# =====================================================================
class QuantumEnv:
    """
    Quantum environment that simulates noisy quantum circuits.
    The environment interacts with the RL agent, providing feedback
    in the form of quantum fidelity.
    """

    def __init__(self, num_qubits, circuit_depth):
        self.backend = AerSimulator()  # Use Qiskit's simulator
        self.noise_model = self._create_noise_model()  # Noise model setup
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
        error = thermal_relaxation_error(t1=50, t2=30, time=10)  # Example noise
        noise_model.add_all_qubit_quantum_error(error, 'u3')
        return noise_model

    def step(self, action):
        # Split action into parameters for U and V circuits
        params_U = action[:len(self.circuit_U.parameters)]
        params_V = action[len(self.circuit_U.parameters):]
        
        # Compute loss using the agent's action
        loss = loss_function(
            matrix=self.matrix, 
            circuit_U=self.circuit_U, 
            circuit_V=self.circuit_V, 
            params_U=params_U, 
            params_V=params_V, 
            rank=self.rank
        )
        
        # Extract estimated singular values based on current parameters
        U = get_unitary(self.circuit_U, params_U)
        V = get_unitary(self.circuit_V, params_V)
        product = np.conj(U.T) @ self.matrix @ V
        _, singular_values_estimated, _ = classical_svd(product)
        estimated_singular_values = singular_values_estimated[:self.rank]
        
        # Compute reward as the negative of loss
        reward = -loss
        
        # Define the next state (could be the estimated singular values)
        state = estimated_singular_values
        
        # Define termination condition (if any)
        done = False  # Modify as needed
        
        # Optionally, include additional info
        info = {}
        
        return state, reward, done, info

    def reset(self):
        # Generate a new random matrix for each episode
        self.matrix = np.random.rand(2**self.num_qubits, 2**self.num_qubits) + \
                      1j * np.random.rand(2**self.num_qubits, 2**self.num_qubits)
        # Return an initial state
        return np.zeros(2 * self.rank)  # Placeholder, adjust as needed

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
        # Define a simple neural network policy with separate outputs for mean and log_std
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean_layer = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable log standard deviation

    def forward(self, state):
        """
        Forward pass through the policy network.
        Input: state (torch.Tensor)
        Output: mean (torch.Tensor)
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean_layer(x)
        return mean

    def select_action(self, state):
        """
        Selects an action based on the current state using the policy.
        Returns both action and log probability for PPO.
        """
        state = torch.tensor(state, dtype=torch.float32)
        mean = self.forward(state)
        std = torch.exp(self.log_std)  # Standard deviation
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -np.pi, np.pi)  # Ensure actions are within desired bounds
        log_prob = dist.log_prob(action).sum()
        return action.detach().numpy(), log_prob.item()

# =====================================================================
# 4. Integrate QSVD into the PPO Workflow
# =====================================================================
def apply_qsvd_and_calibrate(circuit):
    """
    Decomposes the circuit's unitary operator using QSVD and applies
    calibration adjustments to the circuit.
    Input: circuit (QuantumCircuit)
    Output: calibrated_circuit (QuantumCircuit)
    """
    # Placeholder logic for now
    unitary_matrix = np.identity(2)  # Dummy 2x2 matrix for example
    Q, Sigma, V_dagger = qsvd_decompose(unitary_matrix)
    # Adjust the circuit based on decomposed matrices (future work)
    return circuit

# =====================================================================
# 5. Main Training Loop: Agent-Environment Interaction
# =====================================================================
def train_agent(episodes=100, num_qubits=2, circuit_depth=3):
    """
    Trains the PPO agent to adaptively control quantum gates.
    """
    env = QuantumEnv(num_qubits, circuit_depth)
    circuit_U = create_parameterized_circuit(num_qubits, circuit_depth, 'U')
    circuit_V = create_parameterized_circuit(num_qubits, circuit_depth, 'V')
    env.circuit_U = circuit_U
    env.circuit_V = circuit_V
    
    # Calculate the total number of parameters
    total_parameters = len(circuit_U.parameters) + len(circuit_V.parameters)

    # Initialize PPOAgent with the correct action dimension
    agent = PPOAgent(state_dim=2*env.rank, action_dim=total_parameters)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)

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
            memory.append((state, action, log_prob, reward, next_state, done))
            state = next_state
            episode_reward += reward

        print(f"Episode {episode + 1}: Reward = {episode_reward}")

        # Perform PPO update
        if len(memory) > 0:
            optimizer.zero_grad()
            # Collect batches from memory
            states = torch.tensor([m[0] for m in memory], dtype=torch.float32)
            actions = torch.tensor([m[1] for m in memory], dtype=torch.float32)
            old_log_probs = torch.tensor([m[2] for m in memory], dtype=torch.float32)
            rewards = torch.tensor([m[3] for m in memory], dtype=torch.float32)

            # Compute discounted rewards
            discounted_rewards = []
            R = 0
            for r in reversed(rewards.numpy()):
                R = r + gamma * R
                discounted_rewards.insert(0, R)
            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
            # Normalize rewards
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

            # Forward pass
            mean = agent.forward(states)
            std = torch.exp(agent.log_std)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(-1)

            ratios = torch.exp(new_log_probs - old_log_probs)
            advantages = discounted_rewards  # Placeholder: Implement advantage calculation (e.g., GAE)

            # PPO loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().sum(-1).mean()
            loss = policy_loss - entropy_coef * entropy

            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            memory = []  # Clear memory after update

        # Analyze circuit expressiveness and check unitarity every 10 episodes
        if (episode + 1) % 10 == 0:
            analyze_circuit_expressiveness(env.circuit_U, env.circuit_V, 2**num_qubits)
            # Retrieve the latest parameters from the agent
            with torch.no_grad():
                latest_action, _ = agent.select_action(state)
            params_U = latest_action[:len(env.circuit_U.parameters)]
            params_V = latest_action[len(env.circuit_U.parameters):]
            check_unitarity(env.circuit_U, env.circuit_V, params_U, params_V)

# =====================================================================
# 6. Entry Point
# =====================================================================
if __name__ == "__main__":
    """
    Entry point for the program. This is where the training process begins.
    """
    print("Starting training...")
    train_agent(episodes=100, num_qubits=2, circuit_depth=5)  # Increased from 3 to 5
    print("Training complete.")

    # Example usage of run_vqsvd after training
    NUM_QUBITS = 2
    CIRCUIT_DEPTH = 5
    RANK = 2**NUM_QUBITS

    # Generate a random matrix
    M = np.random.rand(2**NUM_QUBITS, 2**NUM_QUBITS) + 1j * np.random.rand(2**NUM_QUBITS, 2**NUM_QUBITS)

    # Perform QSVD using run_vqsvd
    result = run_vqsvd(M, rank=RANK, num_qubits=NUM_QUBITS, circuit_depth=CIRCUIT_DEPTH, lr=0.001, max_iters=100)

    print("QSVD Result:")
    print(result)
