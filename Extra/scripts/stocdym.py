# This file contains code to complement Chapter 3 of "Monte Carlo Methods in Finance" by Jackel
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional
# Class for constructing Weiner processes

class WeinerProcess:
    def __init__(self, T:float, N:int, seed: Optional[int] = None):
        """Initialize the parameters of a Weiner process from t = 0 to t = T at resolution dt = T/N, with seed specification optional."""
        self.T = T
        self.N = N
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates the Weiner process from t = 0 to t = T at resolution dt = T/N."""
        dt: float = self.T / self.N 
        dW: np.ndarray = np.random.normal(0, np.sqrt(dt), self.N)
        path: np.ndarray = np.cumsum(dW)
        time: np.ndarray = np.linspace(0, self.T, self.N + 1)
        return time, np.concatenate(([0], path))

    def transform(self, f: Callable[[np.ndarray], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Applies a function f to the Weiner process path."""
        time, path = self.generate()
        return time, f(path)
    
    def plot(self, f: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> None:
        """Plots the Weiner process, or if f is specified, the process of f applied to the Weiner process."""
        if f is None:
            time, path = self.generate()
            label: str = 'Weiner process path'
            ylabel: str = 'Value'
            title: str = f'Weiner process (Interval: [0, {self.T}], dt = {self.T/self.N})'
        else:
            time, path = self.compose(f)
            label: str = 'Transformed process $X_t = f(W_t)$.' 
            ylabel: str = 'f(X)'
            title: str = f'Transformed Weiner Process $X_t = f(W_t)$ (Interval: [0, {self.T}], dt = {self.T/self.N})'
        plt.plot(time, path, label = label)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel(ylabel)
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        plt.show()