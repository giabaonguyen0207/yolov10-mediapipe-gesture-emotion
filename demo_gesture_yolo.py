import cv2
from ultralytics import YOLO

model = YOLO("models/hand_yolov10_best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.5)
    frame = results[0].plot()

    cv2.imshow("YOLOv10 Hand Gesture", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()