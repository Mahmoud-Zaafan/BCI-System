
__version__ = "1.0.0"
__author__ = "Mahmoud Zaafan"


# ===================================
# src/utils/__init__.py
"""Utility functions for SSVEP analysis."""

from .data_loader import load_ssvep_data, prepare_data, TTAPredictor
from .metrics import evaluate_model, plot_confusion_matrix, calculate_information_transfer_rate
from .visualization import plot_eeg_channels, plot_power_spectrum, plot_class_distribution

__all__ = [
    'load_ssvep_data', 'prepare_data', 'TTAPredictor',
    'evaluate_model', 'plot_confusion_matrix', 'calculate_information_transfer_rate',
    'plot_eeg_channels', 'plot_power_spectrum', 'plot_class_distribution'
]
