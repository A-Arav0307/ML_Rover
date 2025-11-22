from ultralytics import YOLO
import torch
import cv2
cap = cv2.VideoCapture(0)

device = 0
if torch.cuda.is_available():
    device = 0
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU")


model = YOLO("training_results/run1/weights/best.pt")


# result = model("test_image.JPG", device = device)   # replace with actual test image
# result[0].show()


while True:
    ret, frame = cap.read()
    result = model(frame, device = device)   # replace with actual test image
    boxed_frame = result[0].plot()
    cv2.imshow("Image", boxed_frame)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
       break




