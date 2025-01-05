from ultralytics import YOLO

pretrained = "yolo11n.pt"

if __name__ == '__main__':
    augmentations = {
        #"shear":20,
        "translate": 0,
        "scale": 0,
        "hsv_h": 0,
        "hsv_s": 0.3,
        "erasing":0.2,
        "mixup":0.2,
    }

    model = YOLO(pretrained)
    model.train(data='data.yaml', epochs=100, patience=20, device='mps', plots=False, **augmentations) # cache='disk' (default False)

    # # Resume training
    # model = YOLO("path/to/last.pt")  # load a partially trained model
    # results = model.train(resume=True)
