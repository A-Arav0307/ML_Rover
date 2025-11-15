from ultralytics import YOLO
import torch

device = 0
if torch.cuda.is_available():
    device = 0
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU")


model = YOLO("training_results/run1/weights/best.pt")


result = model("test_image.JPG", device = device)   # replace with actual test image
result[0].show()



