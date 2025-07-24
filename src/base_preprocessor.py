"""Base preprocessor class for EEG data."""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    """Abstract base class for EEG preprocessing."""
    
    def __init__(self, fs=250):
        """
        Initialize base preprocessor.
        
        Parameters
        ----------
        fs : int
            Sampling frequency in Hz
        """
        self.fs = fs
        self.nyquist = fs / 2
        
    @abstractmethod
    def preprocess_trial(self, trial_data, *args, **kwargs):
        """Abstract method to be implemented by subclasses."""
        pass
    
    def highpass_filter(self, signal, cutoff, order=4):
        """Apply Butterworth highpass filter."""
        normal_cutoff = cutoff / self.nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, signal)
    
    def lowpass_filter(self, signal, cutoff, order=4):
        """Apply Butterworth lowpass filter."""
        normal_cutoff = cutoff / self.nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)
    
    def bandpass_filter(self, signal, low, high, order=4):
        """Apply Butterworth bandpass filter."""
        low = low / self.nyquist
        high = high / self.nyquist
        
        # Handle edge cases
        if low <= 0:
            low = 0.001
        if high >= 1:
            high = 0.999
            
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)
    
    def notch_filter(self, signal, freq, quality=30):
        """Apply notch filter for line noise removal."""
        b, a = iirnotch(freq, quality, self.fs)
        return filtfilt(b, a, signal)
    
    def remove_dc_offset(self, signal):
        """Remove DC offset from signal."""
        return signal - np.mean(signal)
    
    def robust_normalize(self, signal):
        """Robust normalization using median and MAD."""
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        if mad > 0:
            signal = (signal - median) / (mad * 1.4826)
        return np.clip(signal, -4, 4)
    
    def apply_car(self, data):
        """Apply Common Average Reference."""
        car_signal = np.mean(data, axis=1, keepdims=True)
        return data - car_signal