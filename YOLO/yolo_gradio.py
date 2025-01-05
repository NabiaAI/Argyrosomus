import gradio as gr
from ultralytics import YOLO

model = YOLO("./best.pt")

def predict_image(img, conf_threshold, iou_threshold, show_labels, show_conf):
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=show_labels,
        show_conf=show_conf,
    )
    return results[0].plot() if results else None

iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
        gr.Checkbox(value=True, label="Show Labels"),
        gr.Checkbox(value=True, label="Show Confidence"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Sound ID via YOLO object detection",
    description="Upload images for YOLO object detection.",
    examples=[
        ["./gradio_examples/img1.png", 0.25, 0.45, True, True],
        ["./gradio_examples/img2.png", 0.25, 0.45, True, True],
        ["./gradio_examples/img3.png", 0.25, 0.45, True, True],
    ],
)
iface.launch()