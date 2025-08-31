# This file contains code to complement Chapter 3 of "Monte Carlo Methods in Finance" by Jackel
import numpy as np
import matplotlib.pyplot as plt

# Class for constructing Weiner processes

class WeinerProcess:
    def __init__(self, T, N, seed=None):
        """Initialize the parameters of a Weiner process from t = 0 to t = T at resolution dt = T/N, with seed specification optional."""
        self.T = T
        self.N = N
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

    def generate(self):
        """Generates the Weiner process from t = 0 to t = T at resolution dt = T/N."""
        dt = self.T / self.N 
        dW = np.random.normal(0, np.sqrt(dt), self.N)
        path = np.cumsum(dW)
        time = np.linspace(0, self.T, self.N + 1)
        return time, np.concatenate(([0], path))

    def plot(self):
        """Plots the Weiner process."""
        time, path = self.generate()
        plt.plot(time, path, label='Weiner process path')
        plt.title(f'Weiner process (Interval: [0, {self.T}], dt = {self.T/self.N})')
        plt.xlabel('Time')
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        plt.show()

process = WeinerProcess(1, 100)
plt.figure(figsize=(10, 6))
time, _ = process.generate()  # Get time array once
for _ in range(1000):
    _, path = process.generate()
    plt.plot(time, path, alpha=0.1, color='b')  # Low alpha for visibility
plt.title('1000 Weiner Process Paths (Interval: [0, 1], dt = 0.01)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()