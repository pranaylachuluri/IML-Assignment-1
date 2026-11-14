---
title: Trash Detection
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: mit
short_description: YOLOv8m litter detection, heatmap and cleanliness score
---

# Trash Detection and Cleanliness Score

Upload a photo and a YOLOv8m model fine-tuned on 59 litter classes finds the trash, maps where it is concentrated and rates the scene from 0 (highly polluted) to 100 (clean).

The model runs entirely in your browser with ONNX Runtime Web (WebGPU when available, otherwise WebAssembly), so your photos never leave your device.

Built by Pranay Lachuluri, Odra Bira and Sai Rohith Bharadwaj (University at Buffalo). Code: [github.com/pranaylachuluri/trash-detection-yolov8](https://github.com/pranaylachuluri/trash-detection-yolov8)
