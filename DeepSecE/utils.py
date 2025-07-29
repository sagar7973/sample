import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                            recall_score, roc_auc_score, average_precision_score,
                            confusion_matrix)

def one_hot_encoding(y, num_class=2):
    return np.eye(num_class)[y]

def metrics(y, pred, score):
    """Calculate metrics for binary classification (T6SE vs non-T6SE)"""
    accuracy = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="binary")
    precision = precision_score(y, pred, average="binary", zero_division=0)
    recall = recall_score(y, pred, average="binary", zero_division=0)

    # For binary classification, score should be shape (n_samples, 2)
    if score.shape[1] == 2:  # Binary case with 2 columns
        roc_auc = roc_auc_score(y, score[:, 1])  # Use positive class (T6SE) probability
        average_precision = average_precision_score(y, score[:, 1])
    else:  # Handle edge case if score is 1D
        roc_auc = roc_auc_score(y, score)
        average_precision = average_precision_score(y, score)

    metrics_dict = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "AUC": roc_auc,
        "AUPRC": average_precision
    }
    return metrics_dict

def label2index(label):
    """Convert label to index for binary classification"""
    # If label is already an integer (from our dataset), return as is
    if isinstance(label, (int, np.integer)):
        return label
    
    # If label is string, convert to binary
    if isinstance(label, str):
        if label.startswith('T6SE') or label == '1':
            return 1  # T6SE
        else:
            return 0  # non-T6SE (includes non-secretory, T1SE, T2SE, T3SE, T4SE)
    
    return label

def viz_conf_matrix(cm, labels=['Non-T6SE', 'T6SE'], figsize=(6,5)):
    """Visualize confusion matrix for binary classification"""
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            p = cm_perc[i, j]
            annot[i, j] = f"{cm[i,j]}\n({p:.1f}%)" if p >= 1 else f"{cm[i,j]}"
    
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_df, annot=annot, fmt='', cmap="Blues", ax=ax)
    plt.tight_layout()
    return fig

def print_class_distribution(dataset, dataset_name="Dataset"):
    """Print the class distribution in a dataset"""
    labels = dataset.sequence_labels
    unique, counts = np.unique(labels, return_counts=True)
    
    print(f"\n{dataset_name} class distribution:")
    for label, count in zip(unique, counts):
        class_name = "T6SE" if label == 1 else "Non-T6SE"
        print(f"  {class_name} (label={label}): {count} sequences ({count/len(labels)*100:.1f}%)")
    print(f"  Total: {len(labels)} sequences")

def calculate_sampling_stats(n_t6se_total, n_non_t6se_total, n_t6se_sample, n_non_t6se_sample):
    """Calculate and print sampling statistics"""
    print(f"\nSampling Statistics:")
    print(f"T6SE: {n_t6se_sample}/{n_t6se_total} ({n_t6se_sample/n_t6se_total*100:.1f}%)")
    print(f"Non-T6SE: {n_non_t6se_sample}/{n_non_t6se_total} ({n_non_t6se_sample/n_non_t6se_total*100:.1f}%)")
    
    balance_ratio = n_t6se_sample / n_non_t6se_sample
    print(f"Class balance ratio (T6SE:Non-T6SE) = 1:{1/balance_ratio:.2f}")
    
    return balance_ratio
