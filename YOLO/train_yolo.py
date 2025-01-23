from ultralytics import YOLO
from infer_yolo import YOLOMultiLabelClassifier
import os
import json

pretrained = "yolo11n.pt"

if __name__ == '__main__':
    threshold_val_path = "./data/train/list_valid.txt"
    assert os.path.exists(threshold_val_path), f"Path {threshold_val_path} does not exist"
    train_path = "runs/detect/train"
    if os.path.exists(train_path):
        print(f"Skipping training as model already exists. Please delete {train_path} to train anew.")
        exit()

    augmentations = {
        #"shear":20,
        "translate": 0,
        "scale": 0,
        "hsv_h": 0,
        "hsv_s": 0.3,
        "erasing":0.2,
        "mixup":0.2,
    }
    device = 'mps'

    model = YOLO(pretrained)
    model.train(data='data.yaml', epochs=100, patience=20, device=device, plots=True, imgsz=(64,320), **augmentations) # cache='disk' (default False)

    # # Resume training
    # model = YOLO(f"{train_path}/weights/last.pt")  # load a partially trained model
    # results = model.train(resume=True)

    print("Calculating optimal classification thresholds for best model...")
    model = YOLOMultiLabelClassifier(f"{train_path}/weights", device=device, thresholds=threshold_val_path)
    print("Saving optimal thresholds")
    with open(f"{train_path}/weights/thresholds.json", 'w') as f:
        json.dump(model.thresholds, f)
