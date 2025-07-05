
import cv2
import numpy as np

def segment_hand(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours and len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is not None and len(hull) > 3:
            defects = cv2.convexityDefects(contour, hull)
            if defects is not None:
                s, e, f, d = defects[0][0]
                fingertip = tuple(contour[e][0])
                return mask, contour, fingertip
        return mask, contour, None
    return mask, None, None
