import cv2
import numpy as np

# Define the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera feed
    ret, frame = cap.read()

    # Create a mask for the image
    mask = np.zeros(frame.shape[:2], np.uint8)

    # Define the background and foreground models
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    # Define the rectangle for the object to extract
    rect = (50, 50, 300, 500)

    # Apply GrabCut algorithm to extract the object from the image
    cv2.grabCut(frame, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)

    # Create a new mask where 0 and 2 are converted to 0 and 1 to be used as a mask for the object
    mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

    # Apply the mask to the original image to remove the background
    img = frame * mask2[:, :, np.newaxis]

    # Display the resulting image
    cv2.imshow('Output', img)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
