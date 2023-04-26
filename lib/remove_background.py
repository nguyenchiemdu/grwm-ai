import cv2
import numpy as np
import mediapipe as mp


# Create a background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
# initialize mediapipe
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

def remove_background(bg_image,image):
    # Remove background
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     # Apply the background subtractor to the grayscale frame
    fg_mask = bg_subtractor.apply(gray)

    # Apply a binary threshold to the foreground mask
    threshold = 10
    _, binary_mask = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)

    # Invert the binary mask
    inverted_mask = cv2.bitwise_not(binary_mask)

    # Apply the inverted mask to the original frame to remove the background
    removed_bg = cv2.bitwise_and(image, image, mask=inverted_mask)

    frame = image
    height , width, channel = frame.shape

    frame_with_alpha = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # get the result
    results = selfie_segmentation.process(RGB)

    # extract segmented mask
    mask = results.segmentation_mask
    condition = np.stack(
        (results.segmentation_mask,) * 4, axis=-1) > 0.4

    # resize the background image to the same size of the original frame
    bg_image = cv2.resize(bg_image, (width, height))

    # combine frame and background image using the condition
    output_image = np.where(condition, frame_with_alpha, bg_image)

    return output_image