"""
Data loading utilities for ring-down measurement files.
"""

import numpy as np
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
        # Read the file, skipping comment lines and headers
        data_lines = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if RingDownDataLoader._is_data_line(line):
                    data_lines.append(line)
        
        # Parse data using numpy for better performance
        if not data_lines:
            raise ValueError("No valid data lines found in CSV file")
        
        # Use numpy's genfromtxt-like approach but with manual parsing for better control
        # Split all lines at once and convert to float array
        data_list = []
        for line in data_lines:
            parts = line.split(',')
            # Only take first 4 columns (time and phase are in columns 1 and 4)
            if len(parts) >= 4:
                try:
                    data_list.append([float(parts[0].strip()), float(parts[3].strip())])
                except ValueError:
                    continue  # Skip invalid lines
        
        if not data_list:
            raise ValueError("No valid numeric data found in CSV file")
        
        # Convert to numpy array in one step
        data_array = np.array(data_list)
        
        # Extract time (column 1, index 0) and phase (column 4, index 3)
        # Since we only parsed columns 0 and 3, they're now at indices 0 and 1
        t_raw = data_array[:, 0]
        data_raw = data_array[:, 1]
        
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

