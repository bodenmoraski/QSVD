import numpy as np
from simqrl import SimulatedQuantumEnv, train_agent, plot_training_metrics, CoherentError
from simqsvd import SimulatedQSVD, PerformanceMonitor
import logging
from logging_config import setup_logging
from ppo_agent import PPOAgent
from qiskit_aer import Aer

# Initialize logging with a unique name to prevent duplicate logs
logger = setup_logging(name="simrun")

logging.getLogger('matplotlib.font_manager').disabled = True

def main():
    try:
        logger.info("Starting QSVD training with PPO enhancement...")
        
        # Initialize environment with proper backend setup
        env = SimulatedQuantumEnv(
            num_qubits=4,
            circuit_depth=4,
            noise_params={
                't1': 50e-6,
                't2': 70e-6,
                'thermal_pop': 0.01,
                'single_gate': {'amplitude': 0.001, 'phase': 0.002},
                'two_gate': {'amplitude': 0.002, 'phase': 0.004},
                'readout': {'bit_flip': 0.015}
            }
        )
        
        # Verify backend initialization
        backend = Aer.get_backend('aer_simulator')
        if not backend:
            raise RuntimeError("Failed to initialize quantum backend")
            
        # Initialize and connect agent with improved parameters
        state_dim = env.state_size
        action_dim = env.action_size
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=3e-4,
            gamma=0.99,
            epsilon=0.2,
            c1=1.0,
            c2=0.01
        )
        env.set_agent(agent)
        
        logger.info("Agent initialized and connected to environment")
        
        # Training parameters
        episodes = 1000
        early_stopping_patience = 50
        best_reward = float('-inf')
        episodes_without_improvement = 0
        
        # Training loop with monitoring
        training_metrics = {
            'rewards': [],
            'fidelity_history': [],
            'losses': [],
            'episode_lengths': []
        }
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            while True:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                # Store transition
                agent.store_transition(state, action, reward, next_state, done)
                
                if done:
                    break
                    
                state = next_state
            
            # Update agent
            loss = agent.update()
            
            # Store metrics
            training_metrics['rewards'].append(episode_reward)
            training_metrics['fidelity_history'].append(info.get('fidelity', 0))
            training_metrics['losses'].append(loss)
            training_metrics['episode_lengths'].append(episode_steps)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(training_metrics['rewards'][-10:])
                avg_fidelity = np.mean(training_metrics['fidelity_history'][-10:])
                logger.info(f"Episode {episode + 1}/{episodes}")
                logger.info(f"Average Reward: {avg_reward:.4f}")
                logger.info(f"Average Fidelity: {avg_fidelity:.4f}")
                
                # Early stopping check
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    episodes_without_improvement = 0
                else:
                    episodes_without_improvement += 1
                    
                if episodes_without_improvement >= early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
        
        # Generate final report
        logger.info("Training completed! Generating report...")
        if training_metrics['rewards']:
            logger.info(f"Final Average Reward: {np.mean(training_metrics['rewards'][-100:]):.4f}")
            logger.info(f"Final Average Fidelity: {np.mean(training_metrics['fidelity_history'][-100:]):.4f}")
            logger.info(f"Best Reward Achieved: {best_reward:.4f}")
            
            # Save plot
            plot_training_metrics(training_metrics, save_path='training_results.png')
            
            return agent, training_metrics
        else:
            raise ValueError("No training metrics were collected")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        return None, None
    finally:
        logger.info("Training session ended.")

if __name__ == "__main__":
    logger.info("Starting QSVD program...")
    agent, metrics = main()
    
    if agent is not None and metrics is not None:
        logger.info("Program completed successfully!")
    else:
        logger.error("Program failed to complete successfully")

