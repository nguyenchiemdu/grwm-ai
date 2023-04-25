import cv2
import numpy as np
import random

# Load the pre-trained Mask R-CNN model
# model = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')
model = cv2.ml.SVM_load('svm.xml')

# Define the classes that the model can detect
classes = [
    'background',
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
]

# Define the colors to use for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Prepare the input blob for the model
    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)

    # Set the input blob for the model and run a forward pass
    model.setInput(blob)
    output = model.forward()

    # Loop over the detections
    for i in range(output.shape[2]):
        # Extract the class ID and confidence score for the detection
        class_id = int(output[0, 0, i, 1])
        confidence = output[0, 0, i, 2]

        # Check if the detection is a person and has high enough confidence
        if classes[class_id] == 'person' and confidence > 0.5:
            # Extract the bounding box coordinates
            box = output[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype('int')

            # Extract the mask for the detection
            mask = output[0, 0, i, 5:][0, :, :]

            # Resize the mask to match the size of the frame
            mask = cv2.resize(mask, (endX - startX + 1, endY - startY + 1))
            mask = (mask > 0.3)

            # Apply the mask to the frame to extract the person
            person = frame[startY:endY+1, startX:endX+1][mask]

            # Draw the bounding box and label for the detection
            color = colors[class_id]
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, classes[class_id], (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Apply a blur effect to the background behind the person
            blurred = cv2.GaussianBlur(frame, (51, 51), 0)
            frame[startY:endY+1, startX:endX+1][mask] = blurred[startY:endY+1, startX:endX+1][mask]

    # Show the resulting frame
    cv2.imshow('Human Segmentation', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
