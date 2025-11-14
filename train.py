from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")   # Options: yolov8s.pt, yolov8m.pt, yolov8l.pt

    model.train(
        data="dataset.yaml",      # path to your YAML
        epochs=50,                # adjust for your dataset
        batch=16,
        imgsz=640,
        device=0                  # GPU; change to 'cpu' if needed
    )

    # 3. Evaluate on validation set
    metrics = model.val()
    print("Validation metrics:")
    print(metrics)

    # 4. Run inference on a sample image
    result = model("Hammer\images\5fc43404-IMG_7862.JPG")   # replace with actual test image
    result[0].show()

    # 5. Export the model to ONNX format
    model.export(format="onnx")  # exports best.pt → best.onnx

if __name__ == "__main__":
    main()
