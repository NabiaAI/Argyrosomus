import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

from sklearn.metrics import multilabel_confusion_matrix

sys.path.append('.')
from infer_yolo import YOLOMultiLabelClassifier, load_cached
sys.path.append('macls')
import quantification as quant
#import common

model_path = "YOLO/runs/detect/trainMPS_evenmoredata/weights"
test_list = "CNN/data/test0.txt"
train_list = "CNN/data/train0.txt"
save_matrix_path = "data/output"
bb_threshold = 0.25
np.random.seed(0)


def evaluate_results(all_preds, all_targets, class_labels, model_info):
    # accuacy
    accuracy = (all_preds == all_targets).mean()
    print(f'Accuracy: {accuracy:.5f}')
    #subset accuracy
    subset_accuracy = (all_preds == all_targets).all(axis=1).mean()
    print(f'Subset Accuracy: {subset_accuracy:.5f}')
    # confusion matrix
    cm = multilabel_confusion_matrix(all_targets, all_preds)
    os.makedirs(save_matrix_path, exist_ok=True)
    for i, matrix in enumerate(cm):
        tn, fp, fn, tp = matrix.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        output_string = f'Precision: {precision:.5f}, Recall: {recall:.5f}, Accuracy: {accuracy:.5f}'
        print(f'Label: {class_labels[i]} - {output_string}')

        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', vmin=0, vmax=600, cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        # plt.title(f'Confusion Matrix for Label {class_labels[i]} - {output_string}\n{model_info}')
        try:
            plt.savefig(os.path.join(save_matrix_path, f'yolo_cm_label_{class_labels[i]}.pdf'), bbox_inches='tight', pad_inches=0)
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

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=label_combinations, yticklabels=label_combinations, vmax=300, vmin=0, cbar=False)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if title is not None:
        plt.title(title)
    try:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    except Exception as e:
        print(f'Save confusion martrix{e}')
    plt.close()


if __name__ == '__main__':
    #thresholds =  [0.2650397717952728, 0.5434091091156006, 0.3506118953227997] # train_list

    model = YOLOMultiLabelClassifier(model_path, bounding_box_threshold=bb_threshold)
    spectrograms, labels = load_cached(test_list) 
    preds, probs, boxes = model.predict(spectrograms, save=True, return_boxes=True)

    # filter out negatives (WHICH IS COMPLETELY INADEQUATE)
    # positives_idx =( labels[:, 1] == 1) | (labels[:, 2] == 1)
    # labels = labels[positives_idx][:, 1:]
    # preds = preds[positives_idx][:, 1:]
    # probs = probs[positives_idx][:, 1:]

# RESULT AFTER SCALING FIX: CONFIDENCE MEAN [0.2650397717952728, 0.4205501079559326, 0.32571911811828613]
# Accuracy: 0.90423
# Subset Accuracy: 0.75314
# Label: lt - Precision: 0.86942, Recall: 0.77134, Accuracy: 0.84240
# Label: m - Precision: 0.82700, Recall: 0.90323, Accuracy: 0.91353
# Label: w - Precision: 0.84127, Recall: 0.90598, Accuracy: 0.95676

# RESULT AFTER SCALING FIX: CONFIDENCE MAX [0.2650397717952728, 0.5434091091156006, 0.3506118953227997]
# Accuracy: 0.91167
# Subset Accuracy: 0.76848
# Label: lt - Precision: 0.86942, Recall: 0.77134, Accuracy: 0.84240
# Label: m - Precision: 0.89450, Recall: 0.89862, Accuracy: 0.93724
# Label: w - Precision: 0.86325, Recall: 0.86325, Accuracy: 0.95537
    
    idx = np.arange(len(preds)).reshape((-1, 1))
    np.savetxt('runs/labels.txt', np.hstack((idx, labels)), fmt='%d')
    np.savetxt('runs/preds.txt', np.hstack((idx, preds)), fmt='%d')
    np.savetxt('runs/probs.txt', np.hstack((idx, probs)), fmt='%f')

    which = ['lt', 'm', 'w']
    evaluate_results(preds, labels, which, model_path.split('/')[-3])
    print_combined_multilabel_confusion_matrix("data/output/yolo_multi-label_cm.pdf", labels, preds, list(range(len(which))), which, title=None)
        
    quant.eval_ratio_error(probs, preds, labels, probs, preds, labels, [0,1])
    quant.eval_ratio_error(probs, preds, labels, probs, preds, labels, [1,2])
