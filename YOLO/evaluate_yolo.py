import numpy as np
import sys

sys.path.append('.')
sys.path.append('..')
from infer_yolo import YOLOMultiLabelClassifier, load_cached
sys.path.append('macls')
import quantification as quant
import utils_eval as eval

model_path = "runs/detect/trainMPS_evenmoredata/weights"
train_list = "data/list_train.txt"
save_matrix_path = "../data/output"
bb_threshold = 0.25

def run_on_list(test_list):
    model = YOLOMultiLabelClassifier(model_path, bounding_box_threshold=bb_threshold)
    spectrograms, labels = load_cached(test_list) 
    preds, probs, boxes = model.predict(spectrograms, save=True, return_boxes=True)
    
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


if __name__ == '__main__':
    #thresholds =  [0.2650397717952728, 0.5434091091156006, 0.3506118953227997] # train_list
    test_list = "data/list_valid.txt"
    run_on_list(test_list)