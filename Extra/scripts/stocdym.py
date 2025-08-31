# This file contains code to complement Chapter 3 of "Monte Carlo Methods in Finance" by Jackel
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional
# Class for constructing Weiner processes

class WeinerProcess:
    def __init__(self, T:float, N:int, seed: Optional[int] = None):
        """Initialize the parameters of a Weiner process from t = 0 to t = T at resolution dt = T/N, with seed specification optional."""
        self.T: float = T
        self.N: int = N
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates the Weiner process from t = 0 to t = T at resolution dt = T/N."""
        if self.seed is not None:
            np.random.seed(self.seed)
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
            time, path = self.transform(f)
            label: str = 'Transformed process $X_t = f(W_t)$.' 
            ylabel: str = 'f(X)'
            title: str = f'Transformed Weiner Process $X_t = f(W_t)$ (Interval: [0, {self.T}], dt = {self.T/self.N})'
        plt.plot(time, path, label = label)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel(ylabel)
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        plt.show()

class Vasicek:
    def __init__(self, r0:float, a:float, b:float, sigma: float, T:float, N:int, seed: Optional[int] = None):
        """Initialize the parameters of a Vasicek model r(t) = a(b - r) dt + sigma dW w/ 
        initial value r0,
        long term mean level b, 
        speed of reversion a, 
        and instantaneous volatility sigma, 
        from t = 0 to t = T at resolution dt = T/N, with seed specification optional."""
        self.r0: float = r0
        self.a: float = a
        self.b: float = b
        self.sigma: float = sigma
        self.T: float = T
        self.N: int = N
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates the Vasicek model from t = 0 to t = T at resolution dt = T/N."""
        if self.seed is not None:
            np.random.seed(self.seed)
        dt: float = self.T / self.N
        time: np.ndarray = np.linspace(0, self.T, self.N + 1)
        r: np.ndarray = np.zeros(self.N + 1)
        r[0] = self.r0
        for i in range(1, self.N + 1):
            t: float = i * dt
            z: float = np.random.normal(0, 1)
            r[i] = self.b + (self.r0 - self.b) * np.exp(-self.a * t) + self.sigma * np.sqrt((1 - np.exp(-2 * self.a * t)) / (2 * self.a)) * z
        return time, r
    
    def plot(self) -> None:
        """Plots the Vasicek model"""
        time, path = self.generate()
        label: str = 'Vasicek model path'
        ylabel: str = 'Value'
        title: str = f'Vasicek model $r(t)$ (Interval: [0, {self.T}], dt = {self.T/self.N})'
        plt.plot(time, path, label = label)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel(ylabel)
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        plt.show()

class OrnsteinUhlenbeck:
    pass
        

if __name__ == "__main__":
    process = Vasicek(T=1000, N=1000, r0=5, a=0.2, b=1, sigma=0.02, seed=42)
    process.plot()