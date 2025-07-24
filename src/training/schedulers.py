"""Custom learning rate schedulers."""

import math
import tensorflow as tf


class SGDRScheduler(tf.keras.callbacks.Callback):
    """
    Cosine-annealing learning rate schedule with warm restarts (SGDR).
    
    Reference: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent 
    with Warm Restarts", ICLR 2017.
    """
    
    def __init__(
        self,
        T_0: int = 30,
        T_mult: int = 2,
        eta_max: float = 3e-4,
        eta_min: float = 1e-6,
        verbose: int = 1
    ):
        """
        Initialize SGDR scheduler.
        
        Parameters
        ----------
        T_0 : int
            Number of epochs in the first cosine cycle
        T_mult : int
            Cycle length multiplier
        eta_max : float
            Peak learning rate at the start of each cycle
        eta_min : float
            Minimum learning rate at the end of a cycle
        verbose : int
            Verbosity level (0=silent, 1=line per epoch)
        """
        super().__init__()
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose
        
        # Internal state
        self.T_cur = 0  # Epochs since restart
        self.T_i = T_0  # Current cycle length
        
    @staticmethod
    def _is_tf_variable(x):
        """Check if x is a TensorFlow variable."""
        return isinstance(x, (tf.Variable, tf.Tensor)) and hasattr(x, "assign")
    
    def _set_optimizer_lr(self, lr: float):
        """Safely set the optimizer learning rate."""
        opt = self.model.optimizer
        
        # Try modern attribute first
        if hasattr(opt, "learning_rate"):
            if self._is_tf_variable(opt.learning_rate):
                opt.learning_rate.assign(lr)
            else:
                opt.learning_rate = lr
            return
            
        # Fallback to legacy attribute
        if hasattr(opt, "lr"):
            if self._is_tf_variable(opt.lr):
                opt.lr.assign(lr)
            else:
                opt.lr = lr
                
    def on_epoch_begin(self, epoch, logs=None):
        """Update learning rate at the beginning of each epoch."""
        # Check if we need to restart
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
        
        # Calculate cosine annealing
        cos_inner = math.pi * self.T_cur / self.T_i
        lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (
            1 + math.cos(cos_inner)
        )
        
        # Set the learning rate
        self._set_optimizer_lr(lr)
        self.T_cur += 1
        
        if self.verbose:
            print(f"\nEpoch {epoch+1:03d} — SGDR LR: {lr:.6g}")


class WarmupScheduler(tf.keras.callbacks.Callback):
    """Linear warmup scheduler."""
    
    def __init__(
        self,
        warmup_epochs: int = 5,
        initial_lr: float = 1e-6,
        target_lr: float = 3e-4,
        verbose: int = 0
    ):
        """
        Initialize warmup scheduler.
        
        Parameters
        ----------
        warmup_epochs : int
            Number of warmup epochs
        initial_lr : float
            Initial learning rate
        target_lr : float
            Target learning rate after warmup
        verbose : int
            Verbosity level
        """
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.verbose = verbose
        
    def on_epoch_begin(self, epoch, logs=None):
        """Update learning rate during warmup."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (
                epoch / self.warmup_epochs
            )
            
            # Set the learning rate
            K.set_value(self.model.optimizer.learning_rate, lr)
            
            if self.verbose:
                print(f"\nWarmup Epoch {epoch+1}/{self.warmup_epochs} — LR: {lr:.6g}")


def get_callbacks(config):
    """
    Get training callbacks based on configuration.
    
    Parameters
    ----------
    config : dict
        Training configuration
        
    Returns
    -------
    callbacks : list
        List of callbacks
    """
    callbacks = []
    
    # SGDR scheduler
    if config.get('use_sgdr', True):
        callbacks.append(SGDRScheduler(
            T_0=config.get('sgdr_T0', 30),
            T_mult=config.get('sgdr_Tmult', 2),
            eta_max=config.get('lr_initial', 3e-4),
            eta_min=config.get('lr_min', 1e-6),
            verbose=1
        ))
    
    # Early stopping
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=config.get('early_stopping_patience', 40),
        restore_best_weights=True,
        min_delta=0.001,
        verbose=1
    ))
    
    # Model checkpoint
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ))
    
    # Reduce LR on plateau
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=config.get('reduce_lr_patience', 20),
        min_lr=config.get('lr_min', 1e-7),
        verbose=1
    ))
    
    return callbacks