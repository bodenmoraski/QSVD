import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from qiskit.visualization import plot_bloch_multivector
import os

class QGDVisualizer:
    def __init__(self, output_dir: str = "qgd_visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def plot_cost_history(self, 
                         cost_history: List[float], 
                         title: str = "Cost Function Evolution",
                         save_path: str = None) -> None:
        """
        Plot the evolution of the cost function over iterations.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(cost_history, 'b-', label='Cost')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path))
        plt.show()

    def plot_learning_rate_comparison(self, 
                                    lr_cost_dict: Dict[float, float],
                                    title: str = "Learning Rate vs Final Cost",
                                    save_path: str = None) -> None:
        """
        Plot the relationship between learning rates and final costs.
        """
        learning_rates = list(lr_cost_dict.keys())
        costs = list(lr_cost_dict.values())

        plt.figure(figsize=(10, 6))
        plt.plot(learning_rates, costs, 'ro-')
        plt.xlabel('Learning Rate')
        plt.ylabel('Final Cost')
        plt.title(title)
        plt.grid(True)
        plt.xscale('log')  # Often helpful for learning rate visualization
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path))
        plt.show()

    def plot_statevector_evolution(self,
                                 statevectors: List[np.ndarray],
                                 iterations: List[int],
                                 save_path: str = None) -> None:
        """
        Plot the evolution of quantum states on the Bloch sphere.
        """
        num_states = len(statevectors)
        fig = plt.figure(figsize=(15, 5))
        
        for i, (statevector, iteration) in enumerate(zip(statevectors, iterations)):
            plt.subplot(1, num_states, i + 1)
            plot_bloch_multivector(statevector)
            plt.title(f'Iteration {iteration}')
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path))
        plt.show()

    def plot_parameter_evolution(self,
                               parameter_history: List[List[float]],
                               param_names: List[str] = None,
                               title: str = "Parameter Evolution",
                               save_path: str = None) -> None:
        """
        Plot the evolution of circuit parameters over iterations.
        """
        param_history = np.array(parameter_history)
        num_params = param_history.shape[1]
        
        if param_names is None:
            param_names = [f'θ_{i}' for i in range(num_params)]

        plt.figure(figsize=(10, 6))
        for i in range(num_params):
            plt.plot(param_history[:, i], label=param_names[i])
            
        plt.xlabel('Iteration')
        plt.ylabel('Parameter Value')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path))
        plt.show()

# Example usage:
"""
# With QuantumGradientDescent:
qgd = QuantumGradientDescent(n_qubits=2, target_state=[1, 0, 0, 0])
visualizer = QGDVisualizer()

# After optimization:
final_params, final_cost = qgd.optimize()
visualizer.plot_cost_history(qgd.cost_history, save_path='cost_evolution.png')

# With dynamic epsilon implementation:
lr_cost_results = lr_cost  # From your dynamic epsilon implementation
visualizer.plot_learning_rate_comparison(lr_cost_results, save_path='lr_comparison.png')
"""
