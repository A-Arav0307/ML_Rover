# ML_Rover

Computer vision pipeline for classifying tools (mallet vs hammer) for a rover platform  
(e.g., University Rover Challenge–style tasks that require the robot to identify and interact with specific tools).

The repository contains:

- A YOLO-based object detection model (trained on mallet/hammer images)
- Training and evaluation scripts
- Saved weights and example outputs
- A planned deployment path to **Jetson Nano + TensorRT** for real-time inference on the rover
- So far, we've been able to deploy the CV model locally. 

---

## Repository structure

```text
ML_Rover/
├── Hammer/                 # (If present) Raw/processed images or related assets for hammer class
├── training_results/       # Training logs, metrics, and example outputs (e.g., run1/)
├── weights/                # Saved model weights (e.g., best.pt, last.pt, etc.)
├── dataset.yaml            # YOLO dataset configuration (train/val image paths, class names)
├── evaluate.py             # Script to run inference/evaluation on images or a test set
├── test_image.JPG          # Sample image for quick sanity-check inference
├── train.py                # Script to train the mallet vs hammer model
├── yolo11n.pt              # YOLO11-nano base model checkpoint
├── yolov8n.pt              # YOLOv8-nano base model checkpoint
└── README.md               # This file
