import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2

#path = "/Users/I538904/gitrepos/Argyrosomus/YOLO/data/train/images/Montijo_20210712_93000_segment_409.png" # low weakfish
#path = "YOLO/data/train/images/Montijo_20210712_93000_segment_406.png" # toadfish
path = "YOLO/data/train/images/20170116_1130__segment_3.png" # small weakfish
image = cv2.imread(path, cv2.IMREAD_COLOR_BGR)
# upscale image by factor 2
image = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2))



# model = fasterrcnn_resnet50_fpn(pretrained=True).eval()
# # Convert to tensor and normalize
# image_tensor = torch.tensor(image / 255.0).float()#.unsqueeze(0)
# image_tensor = image_tensor.permute(2, 0, 1)
# print("image_tensor", image_tensor.shape)

# with torch.no_grad():
#     predictions = model([image_tensor])

# for box, score in zip(predictions[0]['boxes'], predictions[0]['scores']):
#     print("box score", score)
#     if score > 0.01:
#         x1, y1, x2, y2 = box.int().tolist()
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# cv2.imshow("RPN Detections", image)
# cv2.waitKey(0)

from segment_anything import SamPredictor, sam_model_registry
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Change to 'sam_vit_l.pth' or 'sam_vit_b.pth' if needed
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

predictor = SamPredictor(sam)
# Load and preprocess spectrogram
predictor.set_image(image)

# Predict segmentation masks
masks, _, _ = predictor.predict()

# Plot the masks
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(image)
for mask in masks:
    segmentation = mask['segmentation']
    color = np.random.rand(3)  # Random color for each mask
    plt.imshow(segmentation, alpha=0.4, cmap="jet")  # Overlay masks

plt.title("Segmented Masks")
plt.axis("off")
plt.show()
