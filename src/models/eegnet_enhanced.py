"""Enhanced EEGNet with Multi-Head Attention for SSVEP."""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def create_attention_block(inputs, num_heads=4):
    """Multi-head attention block for EEG."""
    shape = list(inputs.shape)
    
    if len(shape) == 4:
        _, height, width, features = shape
        if height == 1:  # EEGNet case after DepthwiseConv2D
            time_steps = width
            reshaped = layers.Reshape((time_steps, features))(inputs)
        else:
            time_steps = height
            reshaped = layers.Reshape((time_steps, width * features))(inputs)
    else:
        reshaped = inputs
        time_steps = shape[1]
    
    # Multi-head attention
    attention = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=features // num_heads if features >= num_heads else features,
        dropout=0.1
    )(reshaped, reshaped)
    
    # Add & Norm
    attention = layers.Add()([reshaped, attention])
    attention = layers.LayerNormalization(epsilon=1e-6)(attention)
    
    # Reshape back
    if len(shape) == 4:
        if height == 1:
            output = layers.Reshape((1, time_steps, features))(attention)
        else:
            output = layers.Reshape((time_steps, width, features))(attention)
    else:
        output = attention
    
    return output


def EEGNetEnhanced(
    nb_classes, 
    Chans=8, 
    Samples=1250,
    dropoutRate=0.25, 
    kernLength=64, 
    F1=16, 
    D=2, 
    F2=32,
    use_attention=True
):
    """
    Enhanced EEGNet with attention mechanism for SSVEP.
    
    Parameters
    ----------
    nb_classes : int
        Number of classes
    Chans : int
        Number of EEG channels
    Samples : int
        Number of time samples
    dropoutRate : float
        Dropout rate
    kernLength : int
        Length of temporal convolution kernel
    F1 : int
        Number of temporal filters
    D : int
        Depth multiplier
    F2 : int
        Number of pointwise filters
    use_attention : bool
        Whether to use attention mechanism
        
    Returns
    -------
    model : tf.keras.Model
        Compiled model
    """
    
    input_layer = layers.Input(shape=(Chans, Samples, 1))
    
    # Block 1: Temporal + Spatial Filtering
    block1 = layers.Conv2D(
        F1, (1, kernLength), 
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(0.0005)
    )(input_layer)
    block1 = layers.BatchNormalization()(block1)
    
    # Depthwise convolution
    block1 = layers.DepthwiseConv2D(
        (Chans, 1), 
        use_bias=False,
        depth_multiplier=D,
        depthwise_constraint=tf.keras.constraints.max_norm(1.)
    )(block1)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.Activation('elu')(block1)
    block1 = layers.AveragePooling2D((1, 4))(block1)
    block1 = layers.Dropout(dropoutRate)(block1)
    
    # Add attention after first block
    if use_attention:
        block1 = create_attention_block(block1, num_heads=4)
    
    # Block 2: Separable Convolution
    block2 = layers.SeparableConv2D(
        F2, (1, 16),
        use_bias=False, 
        padding='same',
        depthwise_regularizer=regularizers.l2(0.0005)
    )(block1)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.Activation('elu')(block2)
    block2 = layers.AveragePooling2D((1, 8))(block2)
    block2 = layers.Dropout(dropoutRate)(block2)
    
    # Block 3: Additional Feature Extraction
    block3 = layers.SeparableConv2D(
        F2*2, (1, 8),
        use_bias=False, 
        padding='same'
    )(block2)
    block3 = layers.BatchNormalization()(block3)
    block3 = layers.Activation('elu')(block3)
    
    # Global Average Pooling
    gap = layers.GlobalAveragePooling2D()(block3)
    
    # Dense layers with regularization
    dense = layers.Dense(
        nb_classes*16, 
        activation='elu',
        kernel_regularizer=regularizers.l2(0.001)
    )(gap)
    dense = layers.Dropout(0.3)(dense)
    
    dense2 = layers.Dense(
        nb_classes*8, 
        activation='elu',
        kernel_regularizer=regularizers.l2(0.001)
    )(dense)
    dense2 = layers.Dropout(0.3)(dense2)
    
    # Output layer
    output = layers.Dense(
        nb_classes, 
        activation='softmax',
        kernel_constraint=tf.keras.constraints.max_norm(0.25)
    )(dense2)
    
    return models.Model(inputs=input_layer, outputs=output)