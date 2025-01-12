import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from newsimqsvd import simulate_vqsvd_with_noise

class QSVDNoiseEnv(gym.Env):
    def __init__(self, M, rank, circuit_depth=20):
        super(QSVDNoiseEnv, self).__init__()
        n = M.shape[0]
        self.M = M
        self.rank = rank
        self.circuit_depth = circuit_depth
        
        # Calculate total dimensions
        self.u_dim = n * rank  # 8 * 3 = 24 for U matrix
        self.d_dim = rank      # 3 for diagonal values
        self.v_dim = n * rank  # 8 * 3 = 24 for V matrix
        total_dim = self.u_dim + self.d_dim + self.v_dim  # 24 + 3 + 24 = 51
        
        # Action and observation spaces with explicit dimensions
        self.action_space = spaces.Box(
            low=-0.1,
            high=0.1,
            shape=(total_dim,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim + 1,),  # +1 for initial error
            dtype=np.float32
        )

        # Initial state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Get initial noisy decomposition
        self.U_noisy, self.D_noisy, self.V_noisy, _ = simulate_vqsvd_with_noise(
            self.M, 
            self.rank, 
            circuit_depth=self.circuit_depth
        )
        
        # Calculate and store initial error
        D_diag = np.diag(self.D_noisy)
        M_reconstructed = self.U_noisy @ D_diag @ self.V_noisy.T
        self.initial_frobenius_error = np.linalg.norm(self.M - M_reconstructed, ord='fro') / np.linalg.norm(self.M, ord='fro')
        
        # Enhanced state: Include current error and initial error
        observation = np.concatenate([
            self.U_noisy.flatten(),
            self.D_noisy,
            self.V_noisy.flatten(),
            [self.initial_frobenius_error]  # Add initial error level
        ])
        
        return observation, {}

    def step(self, action):
        n, r = self.M.shape[0], self.rank
        
        # Debug action shape
        action = np.array(action, dtype=np.float32)
        if len(action.shape) == 0:
            action = np.array([action])  # Handle scalar case
        action = action.flatten()
        
        # Verify action dimensions
        expected_dim = self.u_dim + self.d_dim + self.v_dim
        if action.size != expected_dim:
            raise ValueError(f"Action size {action.size} does not match expected size {expected_dim}")
        
        # Scale down actions for more stable learning
        action = action * 0.1
        
        # Calculate indices for slicing
        u_end = self.u_dim
        d_end = u_end + self.d_dim
        v_end = d_end + self.v_dim
        
        # Safely reshape with dimension checks
        try:
            U_adjust = action[:u_end].reshape(n, r)
            D_adjust = action[u_end:d_end]
            V_adjust = action[d_end:v_end].reshape(n, r)
        except ValueError as e:
            print(f"Action shape error: action size = {action.size}, u_end = {u_end}, d_end = {d_end}, v_end = {v_end}")
            print(f"Trying to reshape to: U({n}, {r}), D({self.d_dim},), V({n}, {r})")
            raise e

        self.U_noisy += U_adjust
        self.D_noisy += D_adjust
        self.V_noisy += V_adjust

        # Ensure orthonormality after adjustment
        self.U_noisy, _ = np.linalg.qr(self.U_noisy)
        self.V_noisy, _ = np.linalg.qr(self.V_noisy)

        # Reconstruct the matrix using rank x rank diagonal matrix
        D_diag = np.diag(self.D_noisy)  # Creates r x r diagonal matrix
        M_reconstructed = self.U_noisy @ D_diag @ self.V_noisy.T

        # Calculate all error metrics
        frobenius_error = np.linalg.norm(self.M - M_reconstructed, ord='fro') / np.linalg.norm(self.M, ord='fro')
        
        # Calculate orthonormality error
        orthonormality_error = (
            np.linalg.norm(self.U_noisy.T @ self.U_noisy - np.eye(r), ord='fro') +
            np.linalg.norm(self.V_noisy.T @ self.V_noisy - np.eye(r), ord='fro')
        )
        
        # Quick spectral properties (O(n) operations)
        M_diag = np.diag(self.M)  # Main diagonal elements
        R_diag = np.diag(M_reconstructed)
        
        # Compare diagonal dominance
        diag_similarity = 1 - np.linalg.norm(M_diag - R_diag) / np.linalg.norm(M_diag)
        
        # Combine all components into final reward
        reward = (
            2.0 * diag_similarity     # Diagonal similarity
            - frobenius_error         # Reconstruction error
            - 0.01 * orthonormality_error  # Keep matrices well-formed
        )

        # Next state
        observation = np.concatenate([
            self.U_noisy.flatten(),
            self.D_noisy,
            self.V_noisy.flatten(),
            [self.initial_frobenius_error]
        ])
        
        terminated = False
        truncated = False
        info = {
            'frobenius_error': frobenius_error,
            'diag_similarity': diag_similarity,
            'orthonormality_error': orthonormality_error
        }
        
        return observation, reward, terminated, truncated, info

    def compute_reward(self, M_reconstructed):
        # Original matrix properties
        M_norm = np.linalg.norm(self.M, ord='fro')
        M_trace = np.trace(self.M.T @ self.M)
        
        # Reconstructed matrix properties
        R_norm = np.linalg.norm(M_reconstructed, ord='fro')
        R_trace = np.trace(M_reconstructed.T @ M_reconstructed)
        
        # Compare properties (O(n) operations)
        norm_match = 1 - abs(M_norm - R_norm) / M_norm
        trace_match = 1 - abs(M_trace - R_trace) / M_trace
        
        # Reconstruction error (already computing this)
        error = np.linalg.norm(self.M - M_reconstructed, ord='fro') / M_norm
        
        return norm_match + trace_match - error

# Example usage
if __name__ == "__main__":
    # Create environment with explicit matrix size and rank
    matrix_size = 8
    rank = 3
    
    env = make_vec_env(
        lambda: QSVDNoiseEnv(M=np.random.rand(matrix_size, matrix_size), rank=rank),
        n_envs=4
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Verify environment dimensions
    test_env = QSVDNoiseEnv(M=np.random.rand(matrix_size, matrix_size), rank=rank)
    print(f"Action space shape: {test_env.action_space.shape}")
    print(f"Observation space shape: {test_env.observation_space.shape}")

    # Create PPO agent with tuned hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1
    )

    # Train for longer
    model.learn(total_timesteps=50000)

    # Evaluate
    obs = env.reset()[0]  # VecEnv reset returns (obs, info)
    total_reward = 0
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)  # Add deterministic=True for evaluation
        # Ensure action is properly shaped for vectorized environment
        if len(action.shape) == 1:
            action = action.reshape(1, -1)  # Reshape to (n_envs, action_dim)
        
        obs, rewards, dones, infos = env.step(action)
        total_reward += rewards[0]
        print(f"Step Reward: {rewards[0]:.4f}, Frobenius Error: {infos[0]['frobenius_error']:.4f}")
        
        if dones[0]:
            obs = env.reset()[0]
    
    print(f"Average reward over 100 steps: {total_reward/100:.4f}")