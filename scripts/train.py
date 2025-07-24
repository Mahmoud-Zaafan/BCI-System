#!/usr/bin/env python
"""
Main training script for SSVEP BCI system.

Usage:
    python train.py [--config path/to/config.yaml]
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.train_ssvep import main as train_main


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SSVEP BCI classification model"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/ssvep_config.yaml',
        help='Path to configuration file'
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("="*60)
    print("MTC-AIC3 BCI COMPETITION - SSVEP TRAINING")
    print("="*60)
    print(f"Configuration: {args.config}")
    
    # Check if config exists
    if not os.path.exists(args.config):
        print(f"\nError: Configuration file not found: {args.config}")
        print("Using default configuration...")
        args.config = None
    
    # Run training
    trainer = train_main(args.config)
    
    print("\nTraining completed successfully!")
    print("Models saved:")
    print("  - final_ssvep_model.h5")
    print("  - ssvep_preprocessor.pkl")
    print("  - ssvep_label_encoder.pkl")


if __name__ == "__main__":
    main()