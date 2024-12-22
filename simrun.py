import numpy as np
from simqrl import (
    SimulatedQuantumEnv, 
    QSVDAgent, 
    train_agent, 
    initialize_metrics,
    plot_training_metrics
)
from simqsvd import SimulatedQSVD, PerformanceMonitor

def main():
    try:
        # Define custom noise parameters
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
        
        print("Starting QSVD training with PPO enhancement...")
        
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
        
        # Plot and save training metrics
        plot_training_metrics(training_metrics, save_path='training_results.png')
        
        return agent, training_metrics
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    agent, metrics = main()

