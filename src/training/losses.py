"""Custom loss functions for SSVEP classification."""

import tensorflow as tf
from tensorflow.keras import backend as K


def focal_loss(gamma=2.0, alpha=0.75):
    """
    Focal loss for addressing class imbalance.
    
    Parameters
    ----------
    gamma : float
        Focusing parameter
    alpha : float
        Weighting factor
        
    Returns
    -------
    loss : function
        Focal loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        
        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = alpha * tf.pow(1.0 - y_pred, gamma)
        focal_loss = focal_weight * cross_entropy
        
        return tf.reduce_sum(focal_loss, axis=-1)
    
    return focal_loss_fixed


def focal_loss_with_label_smoothing(
    gamma=2.0, 
    alpha=0.75, 
    label_smoothing=0.05,
    class_weights=None
):
    """
    Focal loss with label smoothing and class weighting.
    
    Parameters
    ----------
    gamma : float
        Focusing parameter
    alpha : float
        Weighting factor
    label_smoothing : float
        Label smoothing factor
    class_weights : dict
        Class weights dictionary
        
    Returns
    -------
    loss : function
        Enhanced focal loss function
    """
    def _loss(y_true, y_pred):
        # Apply label smoothing
        if label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = (1.0 - label_smoothing) * y_true + label_smoothing / num_classes
        
        # Apply class weights if provided
        if class_weights is not None:
            # Convert class weights to tensor
            weights_tensor = tf.constant(
                list(class_weights.values()), 
                dtype=tf.float32
            )
            # Get the class index for each sample
            class_indices = tf.argmax(y_true, axis=1)
            # Get the corresponding weight
            sample_weights = tf.gather(weights_tensor, class_indices)
            sample_weights = tf.expand_dims(sample_weights, 1)
        else:
            sample_weights = 1.0
        
        # Focal loss calculation
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = alpha * tf.pow(1.0 - y_pred, gamma)
        loss = focal_weight * cross_entropy * sample_weights
        
        return tf.reduce_sum(loss, axis=-1)
    
    return _loss


def center_loss(num_classes, alpha=0.5, feat_dim=2):
    """
    Center loss for improving feature discrimination.
    
    Parameters
    ----------
    num_classes : int
        Number of classes
    alpha : float
        Center loss weight
    feat_dim : int
        Feature dimension
        
    Returns
    -------
    loss : function
        Center loss function
    """
    centers = tf.Variable(
        tf.zeros([num_classes, feat_dim]), 
        dtype=tf.float32,
        trainable=False
    )
    
    def _loss(y_true, features):
        # Get labels
        labels = tf.argmax(y_true, axis=1)
        
        # Compute center loss
        centers_batch = tf.gather(centers, labels)
        diff = centers_batch - features
        center_loss_val = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=1))
        
        # Update centers
        unique_labels, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        
        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff
        
        centers.assign_sub(tf.scatter_nd(
            tf.expand_dims(labels, 1),
            diff,
            [num_classes, feat_dim]
        ))
        
        return center_loss_val
    
    return _loss


def combined_loss(
    num_classes,
    gamma=2.0,
    alpha=0.75,
    label_smoothing=0.05,
    center_loss_weight=0.0,
    class_weights=None
):
    """
    Combined focal and center loss.
    
    Parameters
    ----------
    num_classes : int
        Number of classes
    gamma : float
        Focal loss focusing parameter
    alpha : float
        Focal loss weighting factor
    label_smoothing : float
        Label smoothing factor
    center_loss_weight : float
        Weight for center loss component
    class_weights : dict
        Class weights
        
    Returns
    -------
    loss : function
        Combined loss function
    """
    focal = focal_loss_with_label_smoothing(
        gamma=gamma,
        alpha=alpha,
        label_smoothing=label_smoothing,
        class_weights=class_weights
    )
    
    if center_loss_weight > 0:
        center = center_loss(num_classes=num_classes)
        
        def _combined_loss(y_true, y_pred, features=None):
            focal_val = focal(y_true, y_pred)
            if features is not None:
                center_val = center(y_true, features)
                return focal_val + center_loss_weight * center_val
            return focal_val
        
        return _combined_loss
    else:
        return focal