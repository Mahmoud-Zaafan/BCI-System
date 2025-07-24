"""Visualization utilities for SSVEP analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch


def plot_eeg_channels(
    data, 
    fs=250, 
    channel_names=None,
    title="EEG Signals",
    figsize=(12, 8)
):
    """
    Plot multi-channel EEG data.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data (channels, samples)
    fs : int
        Sampling frequency
    channel_names : list, optional
        Channel names
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    n_channels, n_samples = data.shape
    time = np.arange(n_samples) / fs
    
    if channel_names is None:
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]
    
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    # Plot each channel
    for i, (ax, ch_name) in enumerate(zip(axes, channel_names)):
        ax.plot(time, data[i], 'b-', linewidth=0.5)
        ax.set_ylabel(ch_name, rotation=0, labelpad=20)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(np.percentile(data[i], [1, 99]))
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_power_spectrum(
    data,
    fs=250,
    channel_names=None,
    freq_range=(1, 40),
    target_freqs=None,
    title="Power Spectral Density"
):
    """
    Plot power spectrum of EEG channels.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data (channels, samples)
    fs : int
        Sampling frequency
    channel_names : list, optional
        Channel names
    freq_range : tuple
        Frequency range to display
    target_freqs : dict, optional
        Target SSVEP frequencies to highlight
    title : str
        Plot title
    """
    n_channels = data.shape[0]
    
    if channel_names is None:
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]
    
    plt.figure(figsize=(10, 6))
    
    for i, ch_name in enumerate(channel_names):
        # Compute power spectrum
        freqs, psd = welch(data[i], fs=fs, nperseg=fs*2)
        
        # Filter frequency range
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs = freqs[mask]
        psd = psd[mask]
        
        # Plot
        plt.semilogy(freqs, psd, label=ch_name, alpha=0.8)
    
    # Highlight target frequencies
    if target_freqs:
        for name, freq in target_freqs.items():
            plt.axvline(freq, color='red', linestyle='--', alpha=0.5)
            plt.text(freq, plt.ylim()[1]*0.9, name, 
                    rotation=90, ha='right', va='top')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (V²/Hz)')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_class_distribution(labels, class_names=None, title="Class Distribution"):
    """
    Plot class distribution.
    
    Parameters
    ----------
    labels : array-like
        Class labels
    class_names : list, optional
        Class names
    title : str
        Plot title
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique_labels]
    
    plt.figure(figsize=(8, 6))
    
    # Create bar plot
    bars = plt.bar(class_names, counts, color='skyblue', edgecolor='navy')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom')
    
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add percentage labels
    total = sum(counts)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        percentage = count / total * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{percentage:.1f}%', ha='center', va='center',
                color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    feature_importance,
    feature_names=None,
    top_k=20,
    title="Feature Importance"
):
    """
    Plot feature importance.
    
    Parameters
    ----------
    feature_importance : array-like
        Feature importance scores
    feature_names : list, optional
        Feature names
    top_k : int
        Number of top features to display
    title : str
        Plot title
    """
    # Sort features by importance
    indices = np.argsort(feature_importance)[::-1][:top_k]
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(feature_importance))]
    
    top_names = [feature_names[i] for i in indices]
    top_scores = feature_importance[indices]
    
    plt.figure(figsize=(10, 6))
    
    # Create horizontal bar plot
    y_pos = np.arange(len(top_names))
    plt.barh(y_pos, top_scores, color='skyblue', edgecolor='navy')
    
    plt.yticks(y_pos, top_names)
    plt.xlabel('Importance Score')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_prediction_confidence(
    predictions,
    confidences,
    true_labels=None,
    class_names=None,
    title="Prediction Confidence Distribution"
):
    """
    Plot prediction confidence distribution.
    
    Parameters
    ----------
    predictions : array-like
        Predicted class indices
    confidences : array-like
        Prediction confidences
    true_labels : array-like, optional
        True labels for correct/incorrect separation
    class_names : list, optional
        Class names
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 6))
    
    if true_labels is not None:
        # Separate correct and incorrect predictions
        correct_mask = predictions == true_labels
        
        plt.hist(confidences[correct_mask], bins=30, alpha=0.6, 
                label='Correct', color='green', edgecolor='darkgreen')
        plt.hist(confidences[~correct_mask], bins=30, alpha=0.6,
                label='Incorrect', color='red', edgecolor='darkred')
        plt.legend()
    else:
        plt.hist(confidences, bins=30, alpha=0.8,
                color='skyblue', edgecolor='navy')
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    plt.axvline(np.mean(confidences), color='black', linestyle='--',
                label=f'Mean: {np.mean(confidences):.3f}')
    plt.axvline(np.median(confidences), color='gray', linestyle='--',
                label=f'Median: {np.median(confidences):.3f}')
    
    plt.legend()
    plt.tight_layout()
    plt.show()