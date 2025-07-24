"""Main training script for SSVEP classification."""

import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical

from ..ssvep_preprocessor import SSVEPPreprocessor
from ..models.eegnet_enhanced import EEGNetEnhanced
from ..augmentation.ssvep_augmentation import SSVEPAugmentation
from ..utils.data_loader import load_ssvep_data, prepare_data, TTAPredictor
from ..utils.metrics import evaluate_model, plot_confusion_matrix, plot_training_history
from .schedulers import get_callbacks
from .losses import focal_loss_with_label_smoothing


class SSVEPTrainer:
    """SSVEP model trainer."""
    
    def __init__(self, config_path=None):
        """
        Initialize trainer.
        
        Parameters
        ----------
        config_path : str, optional
            Path to configuration file
        """
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
        
        # Initialize components
        self.preprocessor = SSVEPPreprocessor(
            fs=self.config['data']['sampling_rate'],
            config=self.config
        )
        self.augmenter = SSVEPAugmentation(
            fs=self.config['data']['sampling_rate']
        )
        
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None
        
    def _default_config(self):
        """Get default configuration."""
        return {
            'data': {
                'root_dir': '/kaggle/input/mtcaic3',
                'sampling_rate': 250,
                'samples_per_trial': 1750
            },
            'model': {
                'dropout_rate': 0.3,
                'kernel_length': 64,
                'F1': 16,
                'D': 2,
                'F2': 32,
                'use_attention': True
            },
            'training': {
                'batch_size': 64,
                'epochs': 150,
                'validation_split': 0.2,
                'lr_initial': 3e-4,
                'lr_min': 1e-6,
                'use_sgdr': True,
                'sgdr_T0': 30,
                'sgdr_Tmult': 2,
                'early_stopping_patience': 40,
                'reduce_lr_patience': 20
            },
            'augmentation': {
                'enabled': True
            }
        }
    
    def load_and_prepare_data(self):
        """Load and prepare training data."""
        print("\n" + "="*60)
        print("LOADING AND PREPARING DATA")
        print("="*60)
        
        # Load data
        data = load_ssvep_data(
            self.config['data']['root_dir'],
            self.config['data']['samples_per_trial']
        )
        
        # Combine train and validation
        all_data = data['train'] + data['validation']
        
        # Prepare data
        X, y, ids = prepare_data(all_data, self.preprocessor)
        
        # Apply augmentation if enabled
        if self.config['augmentation']['enabled']:
            print("\nApplying data augmentation...")
            X_aug = self.augmenter.apply_all(X, self.preprocessor)
            y_aug = np.tile(y, 5)  # 5x augmentation
            
            # Add channel dimension
            X_aug = X_aug[..., np.newaxis]
            
            print(f"  Augmented data shape: {X_aug.shape}")
            
            return X_aug, y_aug
        else:
            return X[..., np.newaxis], y
    
    def prepare_labels(self, y):
        """Prepare labels for training."""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_encoded),
            y=y_encoded
        )
        
        self.class_weights = dict(enumerate(class_weights))
        self.n_classes = len(self.label_encoder.classes_)
        
        print(f"\nClass information:")
        print(f"  Number of classes: {self.n_classes}")
        print(f"  Classes: {self.label_encoder.classes_}")
        print(f"  Class weights: {self.class_weights}")
        
        return y_categorical
    
    def build_model(self, input_shape):
        """Build the model."""
        print("\nBuilding model...")
        
        self.model = EEGNetEnhanced(
            nb_classes=self.n_classes,
            Chans=input_shape[1],
            Samples=input_shape[2],
            dropoutRate=self.config['model']['dropout_rate'],
            kernLength=self.config['model']['kernel_length'],
            F1=self.config['model']['F1'],
            D=self.config['model']['D'],
            F2=self.config['model']['F2'],
            use_attention=self.config['model']['use_attention']
        )
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(self.config['training']['lr_initial']),
            loss=focal_loss_with_label_smoothing(
                gamma=2.0,
                alpha=0.75,
                label_smoothing=0.0,
                class_weights=self.class_weights
            ),
            metrics=['accuracy']
        )
        
        print(f"  Model parameters: {self.model.count_params():,}")
        
        return self.model
    
    def train(self, X, y):
        """Train the model."""
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config['training']['validation_split'],
            stratify=np.argmax(y, axis=1),
            random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Build model if not already built
        if self.model is None:
            self.build_model(X.shape)
        
        # Get callbacks
        callbacks = get_callbacks(self.config['training'])
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining complete!")
        
        # Evaluate on validation set
        self.evaluate(X_val, y_val)
        
        return self.history
    
    def evaluate(self, X, y):
        """Evaluate the model."""
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        # Get predictions
        y_pred_proba = self.model.predict(X)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y, axis=1)
        
        # Decode labels
        y_true_labels = self.label_encoder.inverse_transform(y_true)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        # Calculate metrics
        metrics = evaluate_model(
            y_true_labels,
            y_pred_labels,
            class_names=self.label_encoder.classes_
        )
        
        # Plot confusion matrix
        plot_confusion_matrix(
            y_true_labels,
            y_pred_labels,
            class_names=self.label_encoder.classes_,
            title='SSVEP Classification - Confusion Matrix'
        )
        
        return metrics
    
    def predict_with_tta(self, X):
        """Predict using Test Time Augmentation."""
        tta_predictor = TTAPredictor(
            self.model,
            self.augmenter,
            temperature=self.config.get('tta', {}).get('temperature', 2.0)
        )
        
        predictions = []
        confidences = []
        
        for signal in X:
            # Remove channel dimension
            signal = signal[:, :, 0]
            
            # Predict
            class_idx, confidence = tta_predictor.predict(signal)
            predictions.append(class_idx)
            confidences.append(confidence)
        
        return np.array(predictions), np.array(confidences)
    
    def save_model(self, filepath='ssvep_model.h5'):
        """Save the trained model."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def save_preprocessor(self, filepath='ssvep_preprocessor.pkl'):
        """Save the preprocessor."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        print(f"Preprocessor saved to {filepath}")
    
    def save_label_encoder(self, filepath='ssvep_label_encoder.pkl'):
        """Save the label encoder."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Label encoder saved to {filepath}")


def main(config_path=None):
    """Main training function."""
    # Initialize trainer
    trainer = SSVEPTrainer(config_path)
    
    # Load and prepare data
    X, y = trainer.load_and_prepare_data()
    
    # Prepare labels
    y_categorical = trainer.prepare_labels(y)
    
    # Train model
    history = trainer.train(X, y_categorical)
    
    # Plot training history
    plot_training_history(history)
    
    # Save model and components
    trainer.save_model('final_ssvep_model.h5')
    trainer.save_preprocessor('ssvep_preprocessor.pkl')
    trainer.save_label_encoder('ssvep_label_encoder.pkl')
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*60)
    
    return trainer


if __name__ == "__main__":
    main()