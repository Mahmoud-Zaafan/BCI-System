"""Data loading utilities for SSVEP."""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_ssvep_data(root_dir, samples_per_trial=1750):
    """
    Load SSVEP data from MTC-AIC3 dataset structure.
    
    Parameters
    ----------
    root_dir : str
        Root directory of the dataset
    samples_per_trial : int
        Number of samples per trial
        
    Returns
    -------
    data_dict : dict
        Dictionary with train, validation, and test data
    """
    def read_split(split_name):
        """Read data for a specific split."""
        csv_path = os.path.join(root_dir, f"{split_name}.csv")
        df = pd.read_csv(csv_path)
        df_ssvep = df[df["task"] == "SSVEP"]
        split_data = []
        
        print(f"Loading {split_name} SSVEP data...")
        
        for idx, row in tqdm(df_ssvep.iterrows(), total=len(df_ssvep)):
            subject = f"{row['subject_id']}"
            session = str(row["trial_session"])
            trial_num = row["trial"]
            label = row["label"] if "label" in row and not pd.isna(row["label"]) else None
            
            eeg_path = os.path.join(
                root_dir, "SSVEP", split_name, 
                subject, session, "EEGdata.csv"
            )
            
            if not os.path.exists(eeg_path):
                print(f"\nWarning: File not found: {eeg_path}")
                continue
                
            eeg_df = pd.read_csv(eeg_path)
            start_idx = (trial_num - 1) * samples_per_trial
            end_idx = start_idx + samples_per_trial
            trial_data = eeg_df.iloc[start_idx:end_idx]
            
            split_data.append({
                "id": row["id"],
                "subject_id": row["subject_id"],
                "trial_session": session,
                "trial": trial_num,
                "label": label,
                "data": trial_data
            })
        
        print(f"  Loaded {len(split_data)} trials")
        return split_data
    
    return {
        "train": read_split("train"),
        "validation": read_split("validation"),
        "test": read_split("test")
    }


def prepare_data(data_list, preprocessor):
    """
    Prepare data for training.
    
    Parameters
    ----------
    data_list : list
        List of trial dictionaries
    preprocessor : object
        Preprocessor instance
        
    Returns
    -------
    X : np.ndarray
        Preprocessed data
    y : np.ndarray
        Labels
    ids : list
        Trial IDs
    """
    X_list = []
    y_list = []
    ids = []
    
    print("Preprocessing data...")
    
    for trial in tqdm(data_list):
        if trial["label"] is None:
            continue
            
        try:
            processed = preprocessor.preprocess_trial(trial["data"])
            X_list.append(processed)
            y_list.append(trial["label"])
            ids.append(trial["id"])
        except Exception as e:
            print(f"  Skipped trial {trial['id']}: {str(e)}")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"  Processed {len(X)} trials")
    print(f"  Data shape: {X.shape}")
    print(f"  Labels: {np.unique(y)}")
    
    return X, y, ids


class TTAPredictor:
    """Test Time Augmentation predictor."""
    
    def __init__(self, model, augmenter, temperature=2.0):
        """
        Initialize TTA predictor.
        
        Parameters
        ----------
        model : tf.keras.Model
            Trained model
        augmenter : object
            Augmentation instance
        temperature : float
            Temperature for softmax scaling
        """
        self.model = model
        self.augmenter = augmenter
        self.temperature = temperature
        
        # Define TTA transformations
        self.tta_transforms = [
            lambda x: x,  # Original
            lambda x: self.augmenter.phase_perturbation(x[np.newaxis])[0],
            lambda x: self.augmenter.amplitude_perturbation(x[np.newaxis])[0],
            lambda x: self.augmenter.random_phase_erasing(x[np.newaxis])[0]
        ]
    
    def softmax_temperature(self, logits, eps=1e-12):
        """Apply temperature-scaled softmax."""
        scaled_logits = np.log(np.clip(logits, eps, 1.0)) / self.temperature
        exp_logits = np.exp(scaled_logits - scaled_logits.max())
        return exp_logits / exp_logits.sum()
    
    def predict(self, signal):
        """
        Predict using Test Time Augmentation.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal (channels, samples)
            
        Returns
        -------
        class_idx : int
            Predicted class index
        confidence : float
            Prediction confidence
        """
        probabilities = []
        
        for transform in self.tta_transforms:
            # Apply transformation
            augmented = transform(signal)
            
            # Add batch and channel dimensions
            augmented = augmented[..., np.newaxis][np.newaxis]
            
            # Get prediction
            pred = self.model.predict(augmented, verbose=0)[0]
            probabilities.append(pred)
        
        # Average probabilities and apply temperature scaling
        mean_proba = self.softmax_temperature(np.mean(probabilities, axis=0))
        
        class_idx = int(mean_proba.argmax())
        confidence = mean_proba.max()
        
        return class_idx, confidence