from ultralytics import YOLO
from yolosplitter import YoloSplitter
import torch

def main():
    model = YOLO("yolov8n.pt")   # Options: yolov8s.pt, yolov8m.pt, yolov8l.pt

    # ys = YoloSplitter(imgFormat=['.jpg'], labelFormat=['.txt'])
    # df = ys.from_yolo_dir(input_dir="Hammer", ratio=(0.8, 0.2, 0.1)) # train, val, test ratio
    # ys.save_split(output_dir="potholes_split") # Saves the split dataset with a data.yaml file

    device = 0
    if torch.cuda.is_available():
        device = 0
        print("Using GPU")
    else:
        device = "cpu"
        print("Using CPU")


    model.train(
        data="dataset.yaml",      # path to your YAML
        epochs=200,                # adjust for your dataset
        batch=16,
        imgsz=640,
        device=device,                  # GPU; change to 'cpu' if needed
        project="training_results",
        name="run1",
        exist_ok=True       # ← allows overwriting
    )

    # 3. Evaluate on validation set
    metrics = model.val()
    print("Validation metrics:")
    print(metrics)

    # 4. Run inference on a sample image
    result = model("test_image.JPG", device = device)   # replace with actual test image
    result[0].show()

    # 5. Export the model to ONNX format
    model.export(format="onnx")  # exports best.pt → best.onnx

if __name__ == "__main__":
    main()
