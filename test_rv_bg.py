import cv2

# Create a background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Apply the background subtractor to the frame
    fg_mask = bg_subtractor.apply(frame)

    # Apply a binary threshold to the foreground mask
    threshold = 10
    _, binary_mask = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)

    # Invert the binary mask
    inverted_mask = cv2.bitwise_not(binary_mask)

    # Apply the inverted mask to the original frame to remove the background
    removed_bg = cv2.bitwise_and(frame, frame, mask=inverted_mask)

    # Show the resulting frame
    cv2.imshow('frame', removed_bg)
    
    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
