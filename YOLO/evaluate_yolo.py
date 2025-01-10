import numpy as np
import sys

sys.path.append('.')
sys.path.append('..')
from infer_yolo import YOLOMultiLabelClassifier, load_cached
sys.path.append('macls')
import quantification as quant
import utils_eval as eval

model_path = "runs/detect/trainMPS_evenmoredata/weights"
test_list = "data/list_valid.txt"
train_list = "data/list_train.txt"
save_matrix_path = "../data/output"
bb_threshold = 0.25


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
    eval.plot_roc_curves(labels, probs, which, "../data/output/yolo_roc.pdf")
    eval.evaluate_results(preds, labels, which, model_path.split('/')[-3], save_matrix_path)
    eval.print_combined_multilabel_confusion_matrix("../data/output/yolo_multi-label_cm.pdf", labels, preds, list(range(len(which))), which, title=None)
        
    quant.eval_ratio_error(probs, preds, labels, probs, preds, labels, [0,1])
    quant.eval_ratio_error(probs, preds, labels, probs, preds, labels, [1,2])
