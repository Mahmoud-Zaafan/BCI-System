"""SSVEP-specific preprocessing pipeline."""

import numpy as np
import pandas as pd
from .base_preprocessor import BasePreprocessor


class SSVEPPreprocessor(BasePreprocessor):
    """Advanced preprocessing pipeline for SSVEP signals."""
    
    def __init__(self, fs=250, config=None):
        """
        Initialize SSVEP preprocessor.
        
        Parameters
        ----------
        fs : int
            Sampling frequency
        config : dict
            Configuration dictionary
        """
        super().__init__(fs)
        
        # Channel configuration
        self.all_channels = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
        self.primary_channels = ['OZ', 'PO7', 'PO8']
        self.secondary_channels = ['PZ', 'CZ']
        self.eeg_indices = list(range(1, 9))
        
        # SSVEP frequencies
        self.target_frequencies = {
            'Forward': 7.0,
            'Backward': 8.0,
            'Left': 10.0,
            'Right': 13.0
        }
        
        # Load config if provided
        if config:
            self._load_config(config)
    
    def _load_config(self, config):
        """Load configuration from dict."""
        if 'channels' in config:
            self.all_channels = config['channels'].get('all', self.all_channels)
        if 'target_frequencies' in config:
            self.target_frequencies = config.get('target_frequencies', self.target_frequencies)
    
    def preprocess_trial(self, trial_data, trial_length_sec=5.0):
        """
        Preprocess SSVEP trial with advanced pipeline.
        
        Parameters
        ----------
        trial_data : array-like
            Raw trial data
        trial_length_sec : float
            Target trial length in seconds
            
        Returns
        -------
        processed_data : np.ndarray
            Preprocessed trial data
        """
        # Skip transition periods (first and last 1 second)
        skip_samples = int(1.0 * self.fs)
        target_samples = int(trial_length_sec * self.fs)
        
        # Extract EEG channels
        eeg_data = self._extract_eeg_channels(trial_data)
        
        # Trim transition periods
        if len(eeg_data) > skip_samples * 2:
            eeg_data = eeg_data[skip_samples:-skip_samples]
        
        # Ensure correct length
        eeg_data = self._adjust_signal_length(eeg_data, target_samples)
        
        # Process each channel
        processed_data = []
        for ch_idx in range(eeg_data.shape[1]):
            signal = eeg_data[:, ch_idx]
            signal = self._advanced_preprocess(signal)
            processed_data.append(signal)
        
        processed_data = np.array(processed_data)
        
        # Apply spatial filtering
        processed_data = self._apply_spatial_filtering(processed_data)
        
        return processed_data
    
    def _extract_eeg_channels(self, trial_data):
        """Extract EEG channels from trial data."""
        if isinstance(trial_data, pd.DataFrame):
            if all(ch in trial_data.columns for ch in self.all_channels):
                return trial_data[self.all_channels].values
            else:
                return trial_data.iloc[:, self.eeg_indices].values
        else:
            return trial_data[:, self.eeg_indices]
    
    def _advanced_preprocess(self, signal):
        """Apply advanced preprocessing to single channel."""
        # Remove DC offset
        signal = self.remove_dc_offset(signal)
        
        # High-pass filter at 1Hz
        signal = self.highpass_filter(signal, 1.0, order=4)
        
        # Notch filters at 50Hz and harmonics
        for freq in [50, 100]:
            if freq < self.nyquist:
                signal = self.notch_filter(signal, freq, quality=30)
        
        # Low-pass filter at 40Hz
        signal = self.lowpass_filter(signal, 40, order=4)
        
        # Advanced artifact rejection
        signal = self._advanced_artifact_rejection(signal)
        
        # Robust normalization
        signal = self.robust_normalize(signal)
        
        return signal
    
    def _advanced_artifact_rejection(self, signal):
        """Remove artifacts using multiple criteria."""
        # Amplitude-based rejection
        q1, q3 = np.percentile(signal, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 2.5 * iqr
        upper_bound = q3 + 2.5 * iqr
        
        # Gradient-based rejection
        gradient = np.gradient(signal)
        grad_threshold = 5 * np.std(gradient)
        
        # Combined rejection
        artifact_mask = (
            (signal < lower_bound) | 
            (signal > upper_bound) | 
            (np.abs(gradient) > grad_threshold)
        )
        
        # Interpolate artifacts
        if np.any(artifact_mask):
            good_indices = np.where(~artifact_mask)[0]
            bad_indices = np.where(artifact_mask)[0]
            if len(good_indices) > 10:
                signal[bad_indices] = np.interp(
                    bad_indices, good_indices, signal[good_indices]
                )
        
        return signal
    
    def _adjust_signal_length(self, signal, target_length):
        """Adjust signal to target length."""
        if len(signal) > target_length:
            return signal[:target_length]
        elif len(signal) < target_length:
            return np.pad(signal, (0, target_length - len(signal)), 'edge')
        return signal
    
    def _apply_spatial_filtering(self, data):
        """Apply spatial filtering including bipolar derivation."""
        # Apply CAR first
        data = self.apply_car(data)
        
        # Add bipolar derivation if we have enough channels
        if data.shape[0] >= 8:
            # OZ-PZ derivation (channels 6 and 4)
            bipolar = data[6] - data[4]
            data = np.vstack([data, bipolar[np.newaxis, :]])
        
        return data