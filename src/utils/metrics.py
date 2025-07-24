"""Performance metrics and evaluation utilities."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(y_true, y_pred, class_names=None):
    """
    Comprehensive model evaluation.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        Class names for display
        
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # Macro and weighted averages
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'f1_scores': f1,
        'support': support,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    # Print classification report
    print("\nClassification Report:")
    print("=" * 60)
    print(classification_report(
        y_true, y_pred, 
        target_names=class_names,
        digits=4
    ))
    
    # Print summary metrics
    print("\nSummary Metrics:")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")
    
    if class_names:
        print("\nPer-Class F1-Scores:")
        for i, (name, score) in enumerate(zip(class_names, f1)):
            print(f"  {name}: {score:.4f}")
    
    return metrics


def plot_confusion_matrix(
    y_true, 
    y_pred, 
    class_names=None,
    title='Confusion Matrix',
    figsize=(8, 6),
    cmap='Blues'
):
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        Class names
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_training_history(history, metrics=['accuracy', 'loss']):
    """
    Plot training history.
    
    Parameters
    ----------
    history : keras.History
        Training history object
    metrics : list
        Metrics to plot
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        # Plot training and validation
        ax.plot(history.history[metric], 
                label=f'Training {metric.capitalize()}',
                linewidth=2)
        ax.plot(history.history[f'val_{metric}'], 
                label=f'Validation {metric.capitalize()}',
                linewidth=2)
        
        ax.set_title(f'Model {metric.capitalize()}', fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print best metrics
    print("\nBest Metrics:")
    print("=" * 40)
    for metric in metrics:
        if metric in history.history:
            train_best = max(history.history[metric]) if 'acc' in metric else min(history.history[metric])
            val_best = max(history.history[f'val_{metric}']) if 'acc' in metric else min(history.history[f'val_{metric}'])
            
            print(f"{metric.capitalize()}:")
            print(f"  Best Training: {train_best:.4f}")
            print(f"  Best Validation: {val_best:.4f}")
            print(f"  Final Training: {history.history[metric][-1]:.4f}")
            print(f"  Final Validation: {history.history[f'val_{metric}'][-1]:.4f}")


def calculate_information_transfer_rate(accuracy, n_classes, trial_duration):
    """
    Calculate Information Transfer Rate (ITR) for BCI.
    
    Parameters
    ----------
    accuracy : float
        Classification accuracy (0-1)
    n_classes : int
        Number of classes
    trial_duration : float
        Trial duration in seconds
        
    Returns
    -------
    itr : float
        Information transfer rate in bits/min
    """
    if accuracy == 0 or accuracy == 1:
        # Handle edge cases
        return 0.0 if accuracy == 0 else np.log2(n_classes) * 60 / trial_duration
    
    # Calculate ITR using Shannon's formula
    itr_per_trial = (
        np.log2(n_classes) + 
        accuracy * np.log2(accuracy) + 
        (1 - accuracy) * np.log2((1 - accuracy) / (n_classes - 1))
    )
    
    # Convert to bits per minute
    itr = itr_per_trial * 60 / trial_duration
    
    return itr