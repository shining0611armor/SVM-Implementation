import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_metrics_and_plot(y_true, y_pred, labels=None):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Calculate precision, recall, and F1-score
    precision = precision_score(y_true, y_pred, average=None, labels=labels)
    recall = recall_score(y_true, y_pred, average=None, labels=labels)
    f1 = f1_score(y_true, y_pred, average=None, labels=labels)

    # Calculate accuracy
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)

    # Print precision, recall, F1-score, and accuracy
    for i in range(len(labels)):
        print(f"Class {labels[i]} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1-score: {f1[i]:.4f}")

    print(f"Accuracy: {accuracy:.4f}")

    # Set custom color map and font size
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='BuGn', xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})

    # Set the title font
    heatmap.set_title('Confusion Matrix', fontdict={'fontsize': 16, 'family': 'serif'})

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 2, 1, 0, 2]) 
    y_pred = np.array([1, 0, 1, 1, 1, 2, 0, 2]) 
    calculate_metrics_and_plot(y_true, y_pred, labels=[0, 1, 2])
