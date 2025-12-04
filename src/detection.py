import cv2
import numpy as np

def detect_craters(gray_img):
    """
    Detect craters from a grayscale image using HoughCircles.
    """
    edges = cv2.Canny(gray_img, 80, 200)

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=100,
        param2=22,
        minRadius=5,
        maxRadius=80
    )

    return circles

def draw_craters(base_bgr, circles):
    """
    Draw crater circles on top of a *color* (BGR) image.
    """
    output = base_bgr.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        for x, y, r in circles:
            cv2.circle(output, (x, y), r, (0, 0, 255), 2)

    return output
