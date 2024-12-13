import numpy as np

class RunningMeanStd:
    """
    Tracks the running mean and standard deviation of a data stream.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)  # Initialize variance to 1
        self.std = np.ones(shape)  # Initialize std to 1
        self.count = 1e-4  # Small initial count to avoid division by zero
        
    def update(self, x):
        """
        Update running statistics with new batch of data
        Args:
            x: numpy array of same shape as self.mean
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if len(x.shape) > 1 else 1
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        # Update mean and variance using parallel algorithm
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.std = np.sqrt(self.var)
        self.count = tot_count
        
    def normalize(self, x):
        """
        Normalize input using current mean and std
        Args:
            x: numpy array of same shape as self.mean
        Returns:
            normalized array
        """
        return (x - self.mean) / (self.std + 1e-8)  # Add small epsilon to avoid division by zero
