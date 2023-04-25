import cv2
import numpy as np


def crop_image(image,listPoint):
    pts = np.array(listPoint)
    # Find the bounding rectangle that encloses the points
    x, y, w, h = cv2.boundingRect(pts)
    image = cv2.flip(image, 1)

    # Crop the image
    crop = image[y:y+h, x:x+w]
    return crop