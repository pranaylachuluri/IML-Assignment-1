# Trash Detection & Cleanliness Analysis with YOLOv8

A litter detector that finds 59 kinds of trash in real-world photos, maps where the litter is concentrated, and rates how clean the scene is on a 0–100 scale, all behind a simple Gradio app.

Course project for Deep Learning at the University at Buffalo, built by **Pranay Lachuluri**, **Odra Bira** and **Sai Rohith Bharadwaj**.

## What it does

1. **Detects** trash objects with a fine-tuned YOLOv8m model (bottles, cans, wrappers, cups, plastic film, cigarettes and 53 more classes).
2. **Localizes** each item with a bounding box and confidence score.
3. **Builds a heatmap** by adding each detection's confidence over its box, so litter hotspots stand out.
4. **Scores cleanliness** from 0 (highly polluted) to 100 (clean), weighted by detection confidence.
5. **Explains** the result in a Gradio assistant: upload an image and get the detections, heatmap, score and a short summary.

## Results

Validation set: 299 images, 962 labeled objects, 59 classes.

| Model | Image size | Epochs | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv8n (baseline) | 640 | 30 | 0.480 | 0.170 | 0.251 | 0.140 | 0.110 |
| YOLOv8s | 960 | 57 | 0.394 | 0.224 | 0.286 | 0.181 | 0.145 |
| **YOLOv8m (final)** | 832 | 86 | 0.413 | 0.194 | 0.264 | **0.183** | **0.158** |

The final model improves mAP@50-95 by **44%** over the YOLOv8n baseline (0.110 → 0.158). Common, well-represented classes do much better than the average: clear plastic bottles reach 0.597 mAP@50 and aerosol cans 0.574. Rare classes with only a handful of examples pull the 59-class average down.

## Dataset

A TACO-style litter dataset converted to YOLO format: 59 classes, 1,200 training and 299 validation images of outdoor scenes (roads, grass, sand, leaves, sidewalks). Objects are often small, partly hidden or similar in color to the background, which makes this a hard detection task.

## Training setup

- YOLOv8m, COCO-pretrained, fine-tuned at 832×832, batch size 4, up to 100 epochs with early stopping (patience 25)
- AdamW (lr 0.0006, weight decay 0.0005) with a cosine learning-rate schedule
- Augmentation: HSV jitter, rotation, translation, scale, shear, horizontal flip, Mosaic (closed for the last 10 epochs) and MixUp
- Best checkpoint selected on validation mAP

## Tech stack

Python · PyTorch · Ultralytics YOLOv8 · OpenCV · NumPy · Pandas · Matplotlib · Gradio · Google Colab (GPU)

## Run the app

```bash
pip install -r app/requirements.txt
python app/app.py
```

Then open the local link Gradio prints. On first run the app downloads the trained YOLOv8m weights (`best.pt`) from Google Drive. To use your own checkpoint, put it at `app/best.pt` or set `TRASH_WEIGHTS=/path/to/best.pt`.

## Train it yourself

The notebook was written for Google Colab with the dataset on Google Drive.

1. Open `odrabira_pranayla_sramavar (1).ipynb` in Colab and switch the runtime to GPU.
2. Point `BASE_DIR` at your copy of the dataset (a folder containing `working_dataset/taco_yolo/data_fixed.yaml`).
3. Run the cells in order: training, validation, heatmap and cleanliness analysis, then the Gradio app.

## Project files

| File | Contents |
| --- | --- |
| `app/app.py` | Standalone Gradio app: detections, heatmap and cleanliness score |
| `app/requirements.txt` | Packages for the app |
| `odrabira_pranayla_sramavar (1).ipynb` | Training, evaluation, heatmap, cleanliness scoring and Gradio app |
| `DL_Final_Report .pdf` | Full project report |
| `odrabira_pranayla_sramavar_deeplearning.pptx` | Presentation slides |
