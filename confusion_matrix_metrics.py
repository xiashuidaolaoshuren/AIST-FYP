"""
Script to calculate accuracy, precision, recall, and F1 score from a confusion matrix
using scikit-learn.
"""

import numpy as np

def calculate_metrics_from_confusion_matrix(cm):
    """
    Calculate accuracy, precision, recall, and F1 from a confusion matrix.
    
    Parameters:
    cm (numpy.ndarray): Confusion matrix (can be 2x2 for binary or nxn for multiclass)
    
    Returns:
    dict: Dictionary containing the calculated metrics
    """
    # For binary classification (2x2 matrix)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # For multiclass, calculate weighted average
    else:
        total = cm.sum()
        tp = np.diag(cm).sum()
        accuracy = tp / total
        
        # Calculate precision and recall per class
        precisions = []
        recalls = []
        
        for i in range(cm.shape[0]):
            tp_i = cm[i, i]
            fp_i = cm[:, i].sum() - tp_i
            fn_i = cm[i, :].sum() - tp_i
            
            precision_i = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0
            recall_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0
            
            precisions.append(precision_i)
            recalls.append(recall_i)
        
        # Weighted average
        class_weights = cm.sum(axis=1) / total
        precision = np.average(precisions, weights=class_weights)
        recall = np.average(recalls, weights=class_weights)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


# Example 1: Binary Classification
print("=" * 50)
print("Example 1: Binary Classification")
print("=" * 50)

# Create a sample confusion matrix (binary: 2x2)
# Format: [[True Negatives, False Positives],
#          [False Negatives, True Positives]]
cm_binary = np.array([[98, 82],
                      [8, 52]])

print("\nConfusion Matrix:")
print(cm_binary)

metrics = calculate_metrics_from_confusion_matrix(cm_binary)
print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1 Score:  {metrics['f1']:.4f}")
