
__version__ = "1.0.0"
__author__ = "Mahmoud Zaafan"


# ===================================
# src/training/__init__.py
"""Training utilities and scripts."""

from .train_ssvep import SSVEPTrainer
from .schedulers import SGDRScheduler, get_callbacks
from .losses import focal_loss, focal_loss_with_label_smoothing

__all__ = [
    'SSVEPTrainer', 'SGDRScheduler', 'get_callbacks',
    'focal_loss', 'focal_loss_with_label_smoothing'
]