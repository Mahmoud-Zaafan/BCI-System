"""Data augmentation techniques for SSVEP signals."""

import numpy as np


class SSVEPAugmentation:
    """Specialized augmentation techniques for SSVEP."""
    
    def __init__(self, fs=250):
        """
        Initialize SSVEP augmentation.
        
        Parameters
        ----------
        fs : int
            Sampling frequency
        """
        self.fs = fs
        
    def phase_perturbation(self, signal, max_shift_ms=50):
        """
        Apply phase perturbation to maintain frequency content.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal (n_trials, n_channels, n_samples)
        max_shift_ms : int
            Maximum shift in milliseconds
            
        Returns
        -------
        augmented : np.ndarray
            Phase-perturbed signal
        """
        max_shift_samples = int(max_shift_ms * self.fs / 1000)
        
        augmented = []
        for trial in signal:
            # Random phase shift
            shift = np.random.randint(-max_shift_samples, max_shift_samples)
            
            # Apply circular shift to each channel
            shifted_trial = np.zeros_like(trial)
            for ch in range(trial.shape[0]):
                shifted_trial[ch] = np.roll(trial[ch], shift)
            
            augmented.append(shifted_trial)
        
        return np.array(augmented)
    
    def amplitude_perturbation(self, signal, scale_range=(0.8, 1.2)):
        """
        Scale amplitude while preserving relative relationships.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        scale_range : tuple
            Min and max scaling factors
            
        Returns
        -------
        augmented : np.ndarray
            Amplitude-scaled signal
        """
        augmented = []
        
        for trial in signal:
            # Random scaling factor
            scale = np.random.uniform(scale_range[0], scale_range[1])
            augmented.append(trial * scale)
        
        return np.array(augmented)
    
    def frequency_masking(self, signal, preprocessor, mask_prob=0.1):
        """
        Mask specific frequency bands.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        preprocessor : object
            Preprocessor with notch_filter method
        mask_prob : float
            Probability of masking each channel
            
        Returns
        -------
        augmented : np.ndarray
            Frequency-masked signal
        """
        augmented = []
        
        for trial in signal:
            masked_trial = trial.copy()
            
            # Randomly mask frequency bands
            for ch in range(trial.shape[0]):
                if np.random.random() < mask_prob:
                    # Apply random notch filter
                    notch_freq = np.random.uniform(6, 20)
                    masked_trial[ch] = preprocessor.notch_filter(
                        masked_trial[ch], notch_freq, quality=5
                    )
            
            augmented.append(masked_trial)
        
        return np.array(augmented)
        
    def random_phase_erasing(self, X, erase_ratio=0.1):
        """
        Random Phase Erasing - randomly erases phase information.
        
        Parameters
        ----------
        X : np.ndarray
            Input signal
        erase_ratio : float
            Ratio of frequencies to erase
            
        Returns
        -------
        augmented : np.ndarray
            Phase-erased signal
        """
        augmented = []
        
        for trial in X:
            erased_trial = trial.copy()
            n_channels, n_samples = trial.shape
            
            # Apply FFT
            fft_trial = np.fft.fft(trial, axis=1)
            magnitude = np.abs(fft_trial)
            phase = np.angle(fft_trial)
            
            # Randomly erase phase information
            for ch in range(n_channels):
                erase_idx = np.random.choice(
                    n_samples,
                    int(n_samples * erase_ratio),
                    replace=False
                )
                # Set phase to random values
                phase[ch, erase_idx] = np.random.uniform(
                    -np.pi, np.pi, len(erase_idx)
                )
            
            # Reconstruct signal
            fft_reconstructed = magnitude * np.exp(1j * phase)
            erased_trial = np.real(np.fft.ifft(fft_reconstructed, axis=1))
            augmented.append(erased_trial)
        
        return np.array(augmented)
    
    def apply_all(self, signal, preprocessor=None):
        """
        Apply all augmentation techniques.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        preprocessor : object, optional
            Preprocessor for frequency masking
            
        Returns
        -------
        augmented : np.ndarray
            Fully augmented dataset (5x original size)
        """
        augmented_list = [signal]  # Original
        
        # Phase perturbation
        augmented_list.append(self.phase_perturbation(signal))
        
        # Amplitude perturbation
        augmented_list.append(self.amplitude_perturbation(signal))
        
        # Frequency masking (if preprocessor available)
        if preprocessor is not None:
            augmented_list.append(
                self.frequency_masking(signal, preprocessor)
            )
        
        # Random phase erasing
        augmented_list.append(self.random_phase_erasing(signal))
        
        return np.vstack(augmented_list)