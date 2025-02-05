import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

from sklearn.metrics import multilabel_confusion_matrix, roc_curve, auc

np.random.seed(0)

def plot_validation_output(dates, sums1, sums2, true_counts, save_path):
    plt.rcdefaults()
    plt.figure(figsize=(12,4))
    plt.rcParams.update({'font.size': 14})
    true_counts = true_counts - 0.5 # to make the plot more readable
    for sums, line_sytle, marker, label in zip([sums1, sums2], [':', '--'], ['v', 'o'], ['YOLO', 'CNN']):
        if sums is None:
            continue

        for num, color in zip([0, 1, 2], ['orange', 'green', 'red']):
            for d, yone, ytwo in zip(dates, sums[:,num], true_counts[:,num]):
                plt.plot([d, d], [yone, ytwo], marker="", color=color, linestyle=line_sytle, linewidth=1)

        plt.scatter(dates, sums[:,0], label=f'lt {label}', marker=marker,  color='orange')
        plt.scatter(dates, sums[:,1], label=f'm {label}', marker=marker, color='green')
        plt.scatter(dates, sums[:,2], label=f'w {label}', marker=marker, color='red')

    plt.scatter(dates, true_counts[:,0], label='lt gt', marker="*",  color='orange')
    plt.scatter(dates, true_counts[:,1], label='m gt', marker="*", color='green')
    plt.scatter(dates, true_counts[:,2], label='w gt', marker="*", color='red')
    plt.xticks(np.arange(len(dates)), dates, rotation=90)
    plt.grid(axis='x', alpha=0.2)
    plt.legend(fontsize='small')
    plt.ylabel('Count')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def plot_roc_curves(y_true, y_probs, labels, path):
    plt.rcdefaults()
    plt.figure(figsize=(6.4/2, 4.8/2))

    for i, (label, color) in enumerate(zip(labels, ['orange', 'green', 'red'], strict=True)):
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr[:-1], tpr[:-1], label=f'{label} (AUC={roc_auc:.2f})', color=color)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=0.5)
    plt.legend(loc="lower right", fontsize='small')
    plt.savefig(path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def evaluate_results(all_preds, all_targets, class_labels, model_info, save_matrix_path, plot=True):
    plt.rcdefaults()
    # accuacy
    accuracy = (all_preds == all_targets).mean()
    print(f'Accuracy: {accuracy:.5f}')
    #subset accuracy
    subset_accuracy = (all_preds == all_targets).all(axis=1).mean()
    print(f'Subset Accuracy: {subset_accuracy:.5f}')
    # confusion matrix
    cm = multilabel_confusion_matrix(all_targets, all_preds)
    if plot: os.makedirs(save_matrix_path, exist_ok=True)
    for i, matrix in enumerate(cm):
        tn, fp, fn, tp = matrix.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        output_string = f'Precision: {precision:.5f}, Recall: {recall:.5f}, F1:{f1:.5f}, Accuracy: {accuracy:.5f}'
        print(f'Label: {class_labels[i]} - {output_string}')

        if not plot: 
            continue
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', vmin=0, vmax=3500, cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        # plt.title(f'Confusion Matrix for Label {class_labels[i]} - {output_string}\n{model_info}')
        try:
            plt.savefig(os.path.join(save_matrix_path, f'{model_info}_cm_label_{class_labels[i]}.pdf'), bbox_inches='tight', pad_inches=0, transparent=True)
        except Exception as e:
            print(f'Error: Save confusion martrix{e}')
        plt.close()

def print_combined_multilabel_confusion_matrix(path, y_true, y_pred, labels_idx, labels, title=None):
    y_true = y_true[:, labels_idx]
    y_pred = y_pred[:, labels_idx]

    label_combinations = list(product([0, 1], repeat=y_true.shape[-1]))
    label_combinations = [''.join([str(digit) for digit in label]) for label in label_combinations]
    combined_cm = np.zeros((len(label_combinations), len(label_combinations)), dtype=int)
    
    for yt, yp in zip(y_true, y_pred):
        yt = ''.join([str(int(label)) for label in yt])
        yp = ''.join([str(int(label)) for label in yp])
        yt_idx = label_combinations.index(yt)
        yp_idx = label_combinations.index(yp)
        combined_cm[yt_idx, yp_idx] += 1

    for i in range(len(label_combinations)):
        new_label = []
        for digit_idx, digit in enumerate(label_combinations[i]):
            if digit == '1':
                new_label.append(labels[digit_idx])
        if len(new_label) == 0:
            new_label = ['-']
        label_combinations[i] = ','.join(new_label)

    plt.rcdefaults()
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=label_combinations, yticklabels=label_combinations, vmax=2800, vmin=0, cbar=False)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if title is not None:
        plt.title(title)
    try:
        plt.savefig(path, bbox_inches='tight', pad_inches=0, transparent=True)
    except Exception as e:
        print(f'Save confusion martrix{e}')
    plt.close()