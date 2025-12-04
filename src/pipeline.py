import cv2
from .enhancement import enhance_image
from .detection import detect_craters, draw_craters

def process_image(path):
    img = cv2.imread(path)
    enhanced = enhance_image(img)
    circles = detect_craters(enhanced)
    output = draw_craters(enhanced, circles)

    cv2.imwrite("enhanced_output.jpg", enhanced)
    cv2.imwrite("craters_output.jpg", output)

    print("Enhanced image saved as enhanced_output.jpg")
    print("Crater overlay saved as craters_output.jpg")