"""
Data loading utilities for ring-down measurement files.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from scipy.io import loadmat
from scipy.signal import detrend


class RingDownDataLoader:
    """
    Loads ring-down measurement data from CSV and MAT files.
    
    Supports Moku:Lab Phasemeter format:
    - CSV: time in column 1, phase in column 4
    - MAT: moku.data structure with time in column 1, phase in column 4
    """
    
    @staticmethod
    def _is_data_line(line: str) -> bool:
        """Check if a line contains numeric data (not a header or comment)."""
        if not line or line.startswith('%'):
            return False
        
        parts = line.split(',')
        if len(parts) < 4:
            return False
        
        try:
            float(parts[0])  # First column should be numeric
            return True
        except ValueError:
            return False
    
    @staticmethod
    def load_csv(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load CSV data file from Moku:Lab Phasemeter.
        
        Optimized version using pandas for fast CSV parsing.
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
        
        Returns:
        --------
        t : np.ndarray
            Time array (s), starting from 0
        data : np.ndarray
            Phase in cycles (detrended)
        """
        # Use pandas for fast CSV parsing
        # Skip comment lines (starting with '%') and header rows
        # Read only columns 0 (time) and 3 (phase)
        try:
            df = pd.read_csv(
                filepath,
                comment='%',  # Skip lines starting with '%'
                header=None,  # No header row
                usecols=[0, 3],  # Only read time (col 0) and phase (col 3)
                dtype=float,
                engine='c',  # Use C engine for better performance
                na_values=[],  # Don't treat any values as NaN
                skipinitialspace=True,  # Skip whitespace after delimiter
            )
        except (pd.errors.EmptyDataError, ValueError) as e:
            raise ValueError(f"No valid data lines found in CSV file: {e}")
        
        if df.empty:
            raise ValueError("No valid data lines found in CSV file")
        
        # Extract time and phase columns
        t_raw = df.iloc[:, 0].values  # Column 0: time
        data_raw = df.iloc[:, 1].values  # Column 1 (was column 3): phase
        
        # Time starts from 0
        t = t_raw - t_raw[0]
        
        # Detrend phase data
        data = detrend(data_raw, type='constant')
        
        return t, data
    
    @staticmethod
    def load_mat(filepath: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Load MAT data file from Moku:Lab Phasemeter.
        
        Parameters:
        -----------
        filepath : str
            Path to MAT file
        
        Returns:
        --------
        t : np.ndarray
            Time array (s), starting from 0
        data : np.ndarray
            Phase in cycles (detrended)
        V2 : np.ndarray or None
            Phase in cycles (detrended) or None if not available
        """
        mat_data = loadmat(filepath)
        
        # Access the moku.data structure
        moku_data = mat_data['moku']['data'][0, 0]
        
        # Extract time (column 1, index 0) and phase (column 4, index 3)
        t_raw = moku_data[:, 0].flatten()
        data_raw = moku_data[:, 3].flatten()  # Column 4 is index 3
        
        # Check if V2 exists (column 9, index 8)
        V2 = None
        if moku_data.shape[1] > 8:
            V2_raw = moku_data[:, 8].flatten()  # Column 9 is index 8
            V2 = detrend(V2_raw, type='constant')
        
        # Time starts from 0
        t = t_raw - t_raw[0]
        
        # Detrend phase data
        data = detrend(data_raw, type='constant')
        
        return t, data, V2
    
    @staticmethod
    def load(filepath: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]:
        """
        Load data file (CSV or MAT) automatically detecting format.
        
        Parameters:
        -----------
        filepath : str
            Path to data file
        
        Returns:
        --------
        t : np.ndarray
            Time array (s), starting from 0
        data : np.ndarray
            Phase in cycles (detrended)
        V2 : np.ndarray or None
            Phase in cycles (detrended) or None if not available
        file_type : str
            'CSV' or 'MAT'
        """
        path = Path(filepath)
        suffix = path.suffix.lower()
        
        if suffix == '.csv':
            t, data = RingDownDataLoader.load_csv(filepath)
            return t, data, None, 'CSV'
        elif suffix == '.mat':
            t, data, V2 = RingDownDataLoader.load_mat(filepath)
            return t, data, V2, 'MAT'
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Expected .csv or .mat")

