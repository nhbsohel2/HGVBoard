
# Hand Gesture Based Virtual Blackboard Using Webcam

This project implements a real-time gesture-controlled virtual blackboard system using a standard webcam.

## Project Overview

The system enables users to write digits (0–9) or perform commands (e.g., draw, erase) using hand gestures in front of a webcam. No external hardware is required—just a webcam and gesture recognition using computer vision and deep learning.

### Key Features

- Real-time hand gesture recognition using webcam
- HSV-based skin segmentation
- Fingertip tracking and contour extraction
- Static and dynamic gesture recognition via CNN / 3D CNN
- Virtual blackboard canvas that displays drawn gestures
- Use of datasets: EgoHands (for hand detection), Jester (for gesture classification)

## Architecture

1. **Hand Detection**:
   - Uses OpenCV for HSV skin color segmentation.
   - Contours are used to detect hand shape.
   - Fingertip is tracked for drawing.

2. **Gesture Classification**:
   - A 3D CNN is trained on the Jester dataset to classify hand motion gestures.
   - The EGO dataset is used for hand presence detection.
   - Network includes Conv2D → MaxPooling → Dropout → Dense → Softmax.

3. **Drawing Mechanism**:
   - When fingertip is detected, tracked movement is visualized on a black screen.
   - Hand gestures simulate the effect of writing digits or controlling the board.

## Folder Structure

```
HandGestureVirtualBoard/
├── model/                    # Trained model (.h5)
├── dataset/                 
│   ├── ego/                 # Hand detection data
│   └── jester/              # Gesture classification data
├── src/                     # Source code
│   ├── detect.py            # Main real-time program
│   ├── train.py             # CNN/3D-CNN training code
│   └── utils.py             # Helper functions (HSV, contours)
├── canvas/                  # Virtual blackboard output images
├── requirements.txt         # Python dependencies
└── README.md                # Full project description
```

## Requirements

- Python 3.x
- OpenCV 3.2.0
- NumPy 1.13.1
- Pandas 0.20.3
- PyTorch 1.9.0
- MXNet 0.11.0

## Datasets

- [Jester Dataset](https://20bn.com/datasets/jester)
- [EgoHands Dataset](http://vision.soic.indiana.edu/projects/egohands/)
- MNIST for digit recognition (optional)

## Accuracy

- ~90.125% gesture recognition accuracy as per paper
- Real-time performance depending on lighting and background quality

## ✍️ How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the real-time system:
```bash
python src/detect.py
```

3. Train the model (if needed):
```bash
python src/train.py
```

---