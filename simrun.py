import numpy as np
from simqrl import SimulatedQuantumEnv, train_agent, plot_training_metrics, CoherentError
from simqsvd import SimulatedQSVD, PerformanceMonitor
import logging
from logging_config import setup_logging
from ppo_agent import PPOAgent

# Initialize logging with a unique name to prevent duplicate logs
logger = setup_logging(name="simrun")

def main():
    try:
        logger.info("Starting QSVD training with PPO enhancement...")
        
        # Initialize environment
        env = SimulatedQuantumEnv(
            num_qubits=4,
            circuit_depth=4,
            noise_params={
                't1': 50e-6,
                't2': 70e-6,
                'thermal_pop': 0.01,
                'single_gate': {'amplitude': 0.001, 'phase': 0.002, 'over_rotation': 0.003},
                'two_gate': {'amplitude': 0.002, 'phase': 0.004, 'cross_talk': 0.003},
                'readout': {'bit_flip': 0.015, 'thermal_noise': 0.01, 'crosstalk': 0.008}
            }
        )
        
        # Initialize PPO agent
        try:
            agent = PPOAgent(env.state_size, env.action_size)
        except AttributeError as e:
            logger.error(f"Failed to initialize PPOAgent: {str(e)}")
            logger.error("Make sure env.state_size and env.action_size are properly defined")
            raise
        
        # Train agent
        agent, training_metrics = train_agent(episodes=1000, env=env)
        
        # Generate report
        logger.info("Training completed! Generating report...")
        if training_metrics['rewards']:  # Check if we have metrics
            logger.info(f"Final Average Reward: {np.mean(training_metrics['rewards'][-100:]):.4f}")
            logger.info(f"Final Average Fidelity: {np.mean(training_metrics['fidelity_history'][-100:]):.4f}")
            
            # Save plot
            plot_training_metrics(training_metrics, save_path='training_results.png')
            
            return agent, training_metrics
        else:
            raise ValueError("No training metrics were collected")
        
    except CoherentError as ce:
        logger.error(f"Coherent error in main: {str(ce)}")
        return None, None
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

