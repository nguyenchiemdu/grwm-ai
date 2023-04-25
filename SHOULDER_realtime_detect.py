import cv2
import mediapipe as mp
import time
import numpy as np

# initialize mediapipe
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

def get_positon(name):
    xl = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['LEFT_'+name]].x * image.shape[1])
    yl = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['LEFT_'+name]].y * image.shape[0])
    xr = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['RIGHT_'+name]].x * image.shape[1])
    yr = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['RIGHT_'+name]].y * image.shape[0])
    return xl,yl,xr,yr

def expand_point(xl,yl,xr,yr,scale = 0.1):
    point_a =  (xl, yl)
    point_b = (xr, yr)
    # Calculate point C
    ab_distance = np.sqrt((point_b[0] - point_a[0])**2 + (point_b[1] - point_a[1])**2)
    bc_distance = ab_distance * scale
    unit_vector_ab = ((point_b[0] - point_a[0]) / ab_distance, (point_b[1] - point_a[1]) / ab_distance)
    point_c = (int(point_b[0] + bc_distance * unit_vector_ab[0]), int(point_b[1] + bc_distance * unit_vector_ab[1]))
    cv2.circle(image,point_c, 5, (0, 0, 255), -1)
    tpm = point_a
    point_a =  point_b
    point_b = tpm
    # Calculate point C
    ab_distance = np.sqrt((point_b[0] - point_a[0])**2 + (point_b[1] - point_a[1])**2)
    bc_distance = ab_distance * scale
    unit_vector_ab = ((point_b[0] - point_a[0]) / ab_distance, (point_b[1] - point_a[1]) / ab_distance)
    point_c = (int(point_b[0] + bc_distance * unit_vector_ab[0]), int(point_b[1] + bc_distance * unit_vector_ab[1]))
    cv2.circle(image,point_c, 5, (0, 0, 255), -1)

def draw_positon(name):
    x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['LEFT_'+name]].x * image.shape[1])
    y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['LEFT_'+name]].y * image.shape[0])
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['RIGHT_'+name]].x * image.shape[1])
    y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['RIGHT_'+name]].y * image.shape[0])
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
def draw_FPS():
    global frame_count
    frame_count += 1
    end_time = time.time()
    fps = frame_count / (end_time - start_time)
    cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
def remove_background():
    # ========
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
     # Show the resulting frame
    # cv2.imshow('frame', removed_bg)
    frame = cv2.flip(image, 1)
    height , width, channel = frame.shape

    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # get the result
    results = selfie_segmentation.process(RGB)

    # extract segmented mask
    mask = results.segmentation_mask
    condition = np.stack(
        (results.segmentation_mask,) * 3, axis=-1) > 0.5

    # resize the background image to the same size of the original frame
    global bg_image
    bg_image = cv2.resize(bg_image, (width, height))

    # combine frame and background image using the condition
    output_image = np.where(condition, frame, bg_image)
    cv2.imshow("Output", output_image)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Create a background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
cap = cv2.VideoCapture(0)
#  calculate FPS 
frame_count= 0
start_time = time.time()
global results

bg_image = cv2.imread('./person.jpeg')

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    # image = cv2.cvtColor(image, cv2.)
    results = pose.process(image)

    if results.pose_landmarks:
        xl,yl,xr,yr =  get_positon('SHOULDER')
        expand_point(xl,yl,xr,yr)
        xl,yl,xr,yr =  get_positon('HIP')
        expand_point(xl,yl,xr,yr,scale=0.35)
        img = cv2.line(image,  (xl, yl), (xr, yr), (0, 255, 0), 1)
        draw_positon('SHOULDER')
        draw_positon('HIP')
        draw_positon('EYE')
        draw_positon('ELBOW')
        draw_positon('WRIST')
    # Draw FPS on the frame
    draw_FPS()
    remove_background()
    
    cv2.imshow('Shoulder Detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
