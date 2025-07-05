
import cv2
import numpy as np
import torch
from src.utils import segment_hand
from src.train_3dcnn import SimpleGestureCNN
import torchvision.transforms as transforms

# Load gesture classifier
gesture_classes = ["draw", "erase", "clear"]
model = SimpleGestureCNN(num_classes=3)
model.load_state_dict(torch.load("./model/gesture_model.pth", map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict_mode(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0
    input_tensor = transform(resized).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return gesture_classes[predicted.item()]

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
cap = cv2.VideoCapture(0)
mode = "draw"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[100:200, 100:200]  # Smaller area for gesture prediction
    gesture_mode = predict_mode(roi)
    mode = gesture_mode

    draw_roi = frame[200:400, 100:500]
    mask, contour, fingertip = segment_hand(draw_roi)

    if fingertip is not None:
        x, y = fingertip[0] + 100, fingertip[1] + 200
        if mode == "draw":
            cv2.circle(canvas, (x, y), 5, (255, 255, 255), -1)
        elif mode == "erase":
            cv2.circle(canvas, (x, y), 30, (0, 0, 0), -1)
        elif mode == "clear":
            canvas[:] = 0

    frame[200:400, 100:500] = cv2.bitwise_and(draw_roi, draw_roi, mask=mask)
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    cv2.putText(combined, f"Mode: {mode}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.rectangle(combined, (100, 100), (200, 200), (0, 255, 0), 2)
    cv2.putText(combined, "Gesture ROI", (100, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("HGVBoard", combined)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
