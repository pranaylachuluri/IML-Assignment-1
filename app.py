"""Gradio app: YOLOv8m trash detection with a litter heatmap and a 0-100 cleanliness score.

Run:  pip install -r app/requirements.txt && python app/app.py
Weights: put the trained best.pt next to this file, set TRASH_WEIGHTS to its path,
or let the app download it from Google Drive on first run.
"""
import os
from pathlib import Path

import cv2
import gdown
import gradio as gr
import numpy as np
import pandas as pd
from ultralytics import YOLO

WEIGHTS = Path(os.environ.get("TRASH_WEIGHTS", Path(__file__).with_name("best.pt")))
DRIVE_ID = os.environ.get("WEIGHTS_DRIVE_ID", "1DQflLCC_im4c7huI4BkL7gTbxEX0gYu-")

if not WEIGHTS.exists():
    gdown.download(id=DRIVE_ID, output=str(WEIGHTS), quiet=False)

model = YOLO(str(WEIGHTS))


def create_trash_heatmap(image, boxes, confidences):
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = map(int, box)
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        heatmap[y1:y2, x1:x2] += float(conf)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 0.65, heatmap_color, 0.35, 0)


def calculate_cleanliness_score(image, boxes, confidences):
    h, w = image.shape[:2]
    if len(boxes) == 0:
        return 100.0
    pollution = 0.0
    for (x1, y1, x2, y2), conf in zip(boxes, confidences):
        area_ratio = max(0, x2 - x1) * max(0, y2 - y1) / (h * w)
        pollution += area_ratio * float(conf) * 1000
    pollution += len(boxes) * 3
    return round(max(0.0, min(100.0, 100 - pollution)), 2)


def get_cleanliness_label(score):
    if score >= 80:
        return "Clean"
    if score >= 50:
        return "Moderately littered"
    if score >= 25:
        return "Polluted"
    return "Highly polluted"


def analyze(input_image, conf=0.25):
    if input_image is None:
        return None, None, "Please upload an image."
    image_rgb = np.array(input_image.convert("RGB"))
    result = model.predict(source=np.ascontiguousarray(image_rgb[..., ::-1]), imgsz=832, conf=conf, iou=0.5, verbose=False)[0]

    boxes, confidences, classes = [], [], []
    if result.boxes is not None:
        for box in result.boxes:
            boxes.append(box.xyxy[0].cpu().numpy())
            confidences.append(float(box.conf[0]))
            classes.append(model.names[int(box.cls[0])])

    detection_img = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    overlay = create_trash_heatmap(image_rgb, boxes, confidences)
    score = calculate_cleanliness_score(image_rgb, boxes, confidences)
    label = get_cleanliness_label(score)

    summary = f"Detected trash objects: {len(boxes)}\nCleanliness score: {score}/100\nCleanliness level: {label}\n\n"
    if not classes:
        summary += "No trash was detected. The area looks clean to the model."
    else:
        summary += "Detected object types:\n"
        for cls, count in pd.Series(classes).value_counts().items():
            summary += f"- {cls}: {count}\n"
        summary += ("\nThe heatmap highlights where litter is concentrated. The score combines "
                    "the number of objects, their size and the model's confidence.")
    return detection_img, overlay, summary


demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Image(type="pil", label="Upload a photo of a scene with litter"),
        gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold"),
    ],
    outputs=[
        gr.Image(type="numpy", label="Detections"),
        gr.Image(type="numpy", label="Litter heatmap"),
        gr.Textbox(label="Summary", lines=14),
    ],
    title="Trash Detection and Cleanliness Score",
    description=("YOLOv8m fine-tuned on 59 litter classes. Upload a photo to see detected trash, "
                 "a heatmap of litter hotspots and a 0–100 cleanliness score. "
                 "[Code on GitHub](https://github.com/pranaylachuluri/trash-detection-yolov8)"),
)

if __name__ == "__main__":
    demo.launch()
