import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from newsimqsvd import simulate_vqsvd_with_noise
import logging
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import pandas as pd

# Set up logging configuration at the top of the file
def setup_logging():
    # Create a unique log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'qsvd_debug_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

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
            low=-1,
            high=1,
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
        # Add immediate rewards for small improvements
        current_error = np.linalg.norm(self.M - M_reconstructed, ord='fro') / np.linalg.norm(self.M, ord='fro')
        if hasattr(self, 'last_error'):
            improvement = self.last_error - current_error
            improvement_reward = np.clip(improvement * 20, -1, 1)  # Increased weight
        else:
            improvement_reward = 0
        self.last_error = current_error
        
        # Penalize extreme actions
        action_penalty = -0.1 * np.mean(np.abs(action))  # New penalty
        
        return improvement_reward + action_penalty

class MetricsTracker:
    def __init__(self, window_size=100, env=None):
        self.metrics = {
            'frobenius_error': [],
            'diag_similarity': [],
            'rewards': [],
            'orthonormality_error': [],
            'action_magnitudes': [],
            'improvement_rate': []
        }
        self.window_size = window_size
        self.rolling_metrics = {k: deque(maxlen=window_size) for k in self.metrics.keys()}
        self.env = env  # Store reference to environment

    def add_metrics(self, info, reward, action):
        # Store raw metrics
        self.metrics['frobenius_error'].append(info['frobenius_error'])
        self.metrics['diag_similarity'].append(info['diag_similarity'])
        self.metrics['rewards'].append(reward)
        self.metrics['orthonormality_error'].append(info['orthonormality_error'])
        self.metrics['action_magnitudes'].append(np.mean(np.abs(action)))
        
        # Calculate improvement rate
        if len(self.metrics['frobenius_error']) > 1:
            improvement = self.metrics['frobenius_error'][-2] - self.metrics['frobenius_error'][-1]
        else:
            improvement = 0
        self.metrics['improvement_rate'].append(improvement)

        # Update rolling metrics
        for k in self.metrics.keys():
            self.rolling_metrics[k].append(self.metrics[k][-1])

    def plot_metrics(self, save_path=None):
        """Generate and save visualization plots"""
        sns.set_style("whitegrid")
        
        # Create a 3x2 subplot figure
        fig, axes = plt.subplots(3, 2, figsize=(15, 20))
        fig.suptitle('QSVD Training Metrics', fontsize=16)

        # 1. Frobenius Error Over Time
        ax = axes[0, 0]
        ax.plot(self.metrics['frobenius_error'], label='Raw Error')
        ax.plot(pd.Series(self.metrics['frobenius_error']).rolling(20).mean(), 
                label='Moving Average', color='red')
        ax.set_title('Frobenius Error Evolution')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Error')
        ax.legend()

        # 2. Rewards Distribution
        ax = axes[0, 1]
        sns.histplot(data=self.metrics['rewards'], ax=ax, bins=30)
        ax.axvline(np.mean(self.metrics['rewards']), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.metrics['rewards']):.4f}')
        ax.set_title('Reward Distribution')
        ax.legend()

        # 3. Action Magnitudes
        ax = axes[1, 0]
        ax.plot(self.metrics['action_magnitudes'])
        ax.set_title('Average Action Magnitude')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Magnitude')

        # 4. Improvement Rate
        ax = axes[1, 1]
        ax.plot(self.metrics['improvement_rate'])
        ax.set_title('Error Improvement Rate')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Improvement')

        # 5. Diagonal Similarity
        ax = axes[2, 0]
        ax.plot(self.metrics['diag_similarity'])
        ax.set_title('Diagonal Similarity')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Similarity')

        # 6. Orthonormality Error
        ax = axes[2, 1]
        ax.plot(self.metrics['orthonormality_error'])
        ax.set_title('Orthonormality Error')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Error')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def save_metrics(self, filepath):
        """Save metrics to CSV file"""
        df = pd.DataFrame(self.metrics)
        df.to_csv(filepath, index=False)

    def plot_matrices(self, step, save_path=None):
        """Plot matrix comparisons if environment is available"""
        if self.env is None:
            return
            
        D_diag = np.diag(self.env.D_noisy)
        M_reconstructed = self.env.U_noisy @ D_diag @ self.env.V_noisy.T
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_matrix_comparison(
            self.env.M, 
            M_reconstructed,
            f'{save_path}_matrices_step_{step}.png' if save_path else None
        )
        plot_singular_values(
            self.env.M, 
            M_reconstructed,
            f'{save_path}_singvals_step_{step}.png' if save_path else None
        )

def plot_matrix_comparison(original_matrix, reconstructed_matrix, save_path=None):
    """Visualize the original and reconstructed matrices side by side"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original matrix
    im1 = ax1.imshow(original_matrix, cmap='viridis')
    ax1.set_title('Original Matrix')
    plt.colorbar(im1, ax=ax1)
    
    # Reconstructed matrix
    im2 = ax2.imshow(reconstructed_matrix, cmap='viridis')
    ax2.set_title('Reconstructed Matrix')
    plt.colorbar(im2, ax=ax2)
    
    # Difference
    difference = np.abs(original_matrix - reconstructed_matrix)
    im3 = ax3.imshow(difference, cmap='viridis')
    ax3.set_title('Absolute Difference')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_singular_values(original_matrix, reconstructed_matrix, save_path=None):
    """Compare singular value distributions"""
    # Compute singular values
    u1, s1, _ = np.linalg.svd(original_matrix)
    u2, s2, _ = np.linalg.svd(reconstructed_matrix)
    
    plt.figure(figsize=(10, 6))
    plt.plot(s1, 'b-', label='Original SVs')
    plt.plot(s2, 'r--', label='Reconstructed SVs')
    plt.yscale('log')
    plt.title('Singular Value Comparison')
    plt.xlabel('Index')
    plt.ylabel('Singular Value (log scale)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()
    
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
        learning_rate=5e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1
    )

    # Train for longer
    model.learn(total_timesteps=50000)

    # Modify the debug_shapes function to use logging
    def debug_shapes(logger, step_num, action, obs, rewards=None, infos=None):
        debug_info = [
            "\n=== Debug Information ===",
            f"Step number: {step_num}",
            f"Action type: {type(action)}",
            f"Action shape: {action.shape if isinstance(action, np.ndarray) else 'Not numpy array'}"
        ]
        
        if isinstance(action, np.ndarray):
            debug_info.append(f"Action content sample: {action[:2]}")
        
        debug_info.extend([
            f"Observation type: {type(obs)}",
            f"Observation shape: {obs.shape if isinstance(obs, np.ndarray) else 'Not numpy array'}"
        ])
        
        if rewards is not None:
            debug_info.extend([
                f"Rewards type: {type(rewards)}",
                f"Rewards shape: {rewards.shape if isinstance(rewards, np.ndarray) else 'Not numpy array'}",
                f"Rewards content: {rewards}"
            ])
        
        if infos is not None:
            debug_info.extend([
                f"Infos type: {type(infos)}",
                f"Infos content: {infos}"
            ])
        
        debug_info.append("=====================\n")
        logger.debug('\n'.join(debug_info))

    # Create test environment for visualization
    test_env = QSVDNoiseEnv(M=np.random.rand(matrix_size, matrix_size), rank=rank)
    metrics_tracker = MetricsTracker(env=test_env)
    
    # Evaluate
    try:
        logger.info("\nStarting evaluation...")
        obs = env.reset()[0]
        logger.info(f"Initial observation shape: {obs.shape}")
        
        total_reward = 0
        for step in range(100):
            logger.info(f"\nStep {step + 1}/100")
            
            try:
                action, _ = model.predict(obs, deterministic=True)
                debug_shapes(logger, step + 1, action, obs)
            except Exception as e:
                logger.error(f"Error during model prediction:")
                logger.error(f"Observation shape when error occurred: {obs.shape}")
                raise e
            
            try:
                if isinstance(action, np.ndarray):
                    logger.debug(f"Original action shape: {action.shape}")
                    
                    if len(action.shape) == 1:
                        action = np.tile(action, (4, 1))
                        logger.debug(f"Action shape after tiling: {action.shape}")
                    elif len(action.shape) == 2 and action.shape[0] != 4:
                        action = action.reshape(4, -1)
                        logger.debug(f"Action shape after reshaping: {action.shape}")
                    else:
                        logger.debug(f"Action shape unchanged: {action.shape}")
                    
                    logger.debug(f"Final action shape before step: {action.shape}")
                else:
                    logger.warning(f"Warning: Action is not numpy array. Type: {type(action)}")
            except Exception as e:
                logger.error("Error during action shape handling:")
                logger.error(f"Action shape when error occurred: {action.shape if isinstance(action, np.ndarray) else type(action)}")
                raise e
            
            try:
                obs, rewards, dones, infos = env.step(action)
                metrics_tracker.add_metrics(infos[0], rewards.mean(), action[0])  # Using first environment's metrics
                debug_shapes(logger, step + 1, action, obs, rewards, infos)
                
                total_reward += rewards.mean()
                logger.info(f"Step {step + 1} Results:")
                logger.info(f"- Reward: {rewards.mean():.4f}")
                logger.info(f"- Frobenius Error: {infos[0]['frobenius_error']:.4f}")
                
                if any(dones):
                    logger.info("Environment reset triggered")
                    obs = env.reset()[0]
                    logger.info(f"New observation shape after reset: {obs.shape}")
                    
            except Exception as e:
                error_info = [
                    "\n=== Error State Debug ===",
                    "Error occurred during environment step:",
                    f"- Action shape: {action.shape if isinstance(action, np.ndarray) else type(action)}",
                    f"- Observation shape: {obs.shape if isinstance(obs, np.ndarray) else type(obs)}"
                ]
                
                if 'rewards' in locals():
                    error_info.append(f"- Rewards shape: {rewards.shape if isinstance(rewards, np.ndarray) else type(rewards)}")
                if 'infos' in locals():
                    error_info.append(f"- Infos type: {type(infos)}")
                
                error_info.extend([
                    f"- Error message: {str(e)}",
                    "======================\n"
                ])
                
                logger.error('\n'.join(error_info))
                raise e

            if step % 20 == 0:  # Generate visualizations every 20 steps
                metrics_tracker.plot_matrices(step, f'qsvd_viz_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    except Exception as e:
        error_state = [
            "\n=== Final Error State ===",
            "Fatal error occurred:",
            f"- Last known action shape: {action.shape if 'action' in locals() and isinstance(action, np.ndarray) else 'Unknown'}",
            f"- Last known observation shape: {obs.shape if 'obs' in locals() and isinstance(obs, np.ndarray) else 'Unknown'}",
            f"- Error type: {type(e)}",
            f"- Error message: {str(e)}",
            "======================\n"
        ]
        logger.error('\n'.join(error_state))
        raise e

    logger.info("\nEvaluation complete!")
    logger.info(f"Average reward over 100 steps: {total_reward/100:.4f}")

    # After training/evaluation, generate and save visualizations
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_tracker.plot_metrics(f'qsvd_metrics_{timestamp}.png')
    metrics_tracker.save_metrics(f'qsvd_metrics_{timestamp}.csv')