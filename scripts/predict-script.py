#!/usr/bin/env python
"""
Prediction script for SSVEP BCI system.

Usage:
    python predict.py [--data_dir path/to/data] [--output predictions.csv]
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import load_ssvep_data, TTAPredictor
from src.training.losses import focal_loss_with_label_smoothing


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate predictions for SSVEP test data"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/kaggle/input/mtcaic3',
        help='Path to data directory'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='final_ssvep_model.h5',
        help='Path to trained model'
    )
    parser.add_argument(
        '--preprocessor',
        type=str,
        default='ssvep_preprocessor.pkl',
        help='Path to preprocessor'
    )
    parser.add_argument(
        '--label_encoder',
        type=str,
        default='ssvep_label_encoder.pkl',
        help='Path to label encoder'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='ssvep_predictions.csv',
        help='Output CSV file'
    )
    parser.add_argument(
        '--use_tta',
        action='store_true',
        help='Use Test Time Augmentation'
    )
    return parser.parse_args()


def main():
    """Main prediction function."""
    args = parse_args()
    
    print("="*60)
    print("MTC-AIC3 BCI COMPETITION - SSVEP PREDICTION")
    print("="*60)
    
    # Load model and components
    print("\nLoading model and components...")
    
    # Load model with custom objects
    model = load_model(
        args.model,
        custom_objects={
            'focal_loss_with_label_smoothing': focal_loss_with_label_smoothing
        }
    )
    print(f"  Model loaded from {args.model}")
    
    # Load preprocessor
    with open(args.preprocessor, 'rb') as f:
        preprocessor = pickle.load(f)
    print(f"  Preprocessor loaded from {args.preprocessor}")
    
    # Load label encoder
    with open(args.label_encoder, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"  Label encoder loaded from {args.label_encoder}")
    
    # Load test data
    print("\nLoading test data...")
    data = load_ssvep_data(args.data_dir)
    test_data = data['test']
    print(f"  Loaded {len(test_data)} test trials")
    
    # Process test data
    print("\nProcessing test data...")
    X_test = []
    ids = []
    
    for trial in tqdm(test_data):
        ids.append(trial['id'])
        try:
            processed = preprocessor.preprocess_trial(trial['data'])
            X_test.append(processed)
        except Exception as e:
            print(f"\n  Warning: Failed to process trial {trial['id']}: {e}")
            # Add zeros as fallback
            X_test.append(np.zeros((9, 1250)))
    
    X_test = np.array(X_test)
    print(f"  Processed data shape: {X_test.shape}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    
    if args.use_tta:
        print("  Using Test Time Augmentation...")
        from src.augmentation.ssvep_augmentation import SSVEPAugmentation
        
        augmenter = SSVEPAugmentation(fs=250)
        tta_predictor = TTAPredictor(model, augmenter, temperature=2.0)
        
        predictions = []
        confidences = []
        
        for i, signal in enumerate(tqdm(X_test)):
            class_idx, confidence = tta_predictor.predict(signal)
            predictions.append(class_idx)
            confidences.append(confidence)
        
        predictions = np.array(predictions)
        confidences = np.array(confidences)
    else:
        # Standard prediction
        X_test_input = X_test[..., np.newaxis]
        y_pred_proba = model.predict(X_test_input, verbose=1)
        predictions = np.argmax(y_pred_proba, axis=1)
        confidences = np.max(y_pred_proba, axis=1)
    
    # Decode predictions
    labels = label_encoder.inverse_transform(predictions)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': ids,
        'label': labels,
        'confidence': confidences
    })
    
    # Sort by ID
    submission_df = submission_df.sort_values('id')
    
    # Save predictions
    submission_df.to_csv(args.output, index=False)
    print(f"\nPredictions saved to {args.output}")
    
    # Print statistics
    print("\nPrediction Statistics:")
    print("="*40)
    print(f"Total predictions: {len(submission_df)}")
    print(f"Average confidence: {confidences.mean():.4f}")
    print(f"Min confidence: {confidences.min():.4f}")
    print(f"Max confidence: {confidences.max():.4f}")
    
    print("\nClass distribution:")
    for label, count in submission_df['label'].value_counts().items():
        print(f"  {label}: {count}")
    
    print("\nPrediction complete!")


if __name__ == "__main__":
    main()