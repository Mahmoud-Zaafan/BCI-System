# ===================================
# src/models/__init__.py

__version__ = "1.0.0"
__author__ = "Mahmoud Zaafan"

"""Model architectures for SSVEP classification."""

from .eegnet_enhanced import EEGNetEnhanced, create_attention_block

__all__ = ['EEGNetEnhanced', 'create_attention_block']

