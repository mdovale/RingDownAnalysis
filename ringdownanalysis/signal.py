"""
Ring-down signal generation and parameter management.
"""

import numpy as np
from typing import Optional


class RingDownSignal:
    """
    Encapsulates ring-down signal parameters and generation.
    
    A ring-down signal is an exponentially decaying sinusoid:
    x(t) = A0 * exp(-t/tau) * cos(2*pi*f0*t + phi0) + noise
    
    Attributes:
    -----------
    f0 : float
        Frequency (Hz)
    fs : float
        Sampling frequency (Hz)
    N : int
        Number of samples
    A0 : float
        Initial amplitude
    snr_db : float
        Initial signal-to-noise ratio (dB)
    Q : float
        Quality factor
    tau : float
        Decay time constant (s), computed from Q and f0
    sigma : float
        Noise standard deviation, computed from SNR
    """
    
    def __init__(
        self,
        f0: float,
        fs: float,
        N: int,
        A0: float = 1.0,
        snr_db: float = 60.0,
        Q: float = 10000.0,
    ):
        """
        Initialize ring-down signal parameters.
        
        Parameters:
        -----------
        f0 : float
            Frequency (Hz)
        fs : float
            Sampling frequency (Hz)
        N : int
            Number of samples
        A0 : float
            Initial amplitude (default: 1.0)
        snr_db : float
            Initial signal-to-noise ratio in dB (default: 60.0)
        Q : float
            Quality factor (default: 10000.0)
        """
        if f0 <= 0:
            raise ValueError("f0 must be positive")
        if fs <= 0:
            raise ValueError("fs must be positive")
        if N <= 0:
            raise ValueError("N must be positive")
        if A0 <= 0:
            raise ValueError("A0 must be positive")
        if Q <= 0:
            raise ValueError("Q must be positive")
        
        self.f0 = float(f0)
        self.fs = float(fs)
        self.N = int(N)
        self.A0 = float(A0)
        self.snr_db = float(snr_db)
        self.Q = float(Q)
        
        # Compute derived parameters
        self.tau = self.Q / (np.pi * self.f0)
        rho0 = 10.0 ** (self.snr_db / 10.0)
        self.sigma = np.sqrt((self.A0**2 / 2.0) / rho0)
        
        # Signal arrays (generated on demand)
        self._t = None
        self._x = None
        self._phi0 = None
    
    @property
    def t(self) -> np.ndarray:
        """Time array (s)."""
        if self._t is None:
            self._t = np.arange(self.N) / self.fs
        return self._t
    
    @property
    def T(self) -> float:
        """Total observation time (s)."""
        return self.N / self.fs
    
    def generate(
        self,
        phi0: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Generate a noisy ring-down signal.
        
        Parameters:
        -----------
        phi0 : float, optional
            Initial phase (rad). If None, random phase is used.
        rng : np.random.Generator, optional
            Random number generator. If None, default RNG is used.
        
        Returns:
        --------
        t : np.ndarray
            Time array (s)
        x : np.ndarray
            Noisy signal
        phi0 : float
            Initial phase used (rad)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if phi0 is None:
            phi0 = rng.uniform(-np.pi, np.pi)
        
        self._phi0 = phi0
        t = self.t
        
        # Generate ring-down signal: A(t) = A0 * exp(-t/tau)
        A_t = self.A0 * np.exp(-t / self.tau)
        s = A_t * np.cos(2.0 * np.pi * self.f0 * t + phi0)
        w = rng.normal(0.0, self.sigma, size=self.N)
        x = s + w
        
        self._x = x
        return t, x, phi0
    
    def get_signal(self) -> Optional[np.ndarray]:
        """Get the generated signal if available."""
        return self._x
    
    def get_phase(self) -> Optional[float]:
        """Get the initial phase if signal was generated."""
        return self._phi0

