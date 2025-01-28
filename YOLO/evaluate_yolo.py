import numpy as np
import sys
import os

sys.path.append('.')
sys.path.append('..')
from infer_yolo import YOLOMultiLabelClassifier, load_cached
sys.path.append('macls')
import quantification as quant
import utils_eval as eval

model_path = "final_model/weights"
device = 'mps'
save_examples = True
bb_threshold = 0.25
thresholds = None # None = use saved thresholds # [0.25585734844207764, 0.6342101693153381, 0.28556814789772034] [0.2535114586353302, 0.2733498811721802, 0.28548818826675415]

def run_on_list(test_list, output_path):
    model = YOLOMultiLabelClassifier(model_path, bounding_box_threshold=bb_threshold, thresholds=thresholds, device=device)
    spectrograms, labels = load_cached(test_list) 
    preds, probs, boxes = model.predict(spectrograms, save=save_examples, return_boxes=True)
    
    idx = np.arange(len(preds)).reshape((-1, 1))
    np.savetxt('runs/labels.txt', np.hstack((idx, labels)), fmt='%d')
    np.savetxt('runs/preds.txt', np.hstack((idx, preds)), fmt='%d')
    np.savetxt('runs/probs.txt', np.hstack((idx, probs)), fmt='%f')

    which = ['lt', 'm', 'w']
    eval.plot_roc_curves(labels, probs, which, os.path.join(output_path, "yolo_roc.pdf"))
    eval.evaluate_results(preds, labels, which, "yolo", output_path)
    eval.print_combined_multilabel_confusion_matrix(os.path.join(output_path, "yolo_multi-label_cm.pdf"), labels, preds, list(range(len(which))), which, title=None)
        
    quant.eval_ratio_error(probs, preds, labels, probs, preds, labels, [[0,1], [1,2]])


if __name__ == '__main__':
    print("Evaluating YOLO model on validation split of training data")
    test_list = "data/train/list_valid.txt"
    run_on_list(test_list, "../data/output")

    print("Evaluating YOLO model on external validation data")
    test_list = "data/validation/list.txt"
    run_on_list(test_list, "../data/output/validation")