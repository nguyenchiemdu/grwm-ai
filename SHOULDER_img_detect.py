import cv2
import mediapipe as mp
import numpy as np


def get_positon(name):
    xl = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['LEFT_'+name]].x * image.shape[1])
    yl = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['LEFT_'+name]].y * image.shape[0])
    xr = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['RIGHT_'+name]].x * image.shape[1])
    yr = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['RIGHT_'+name]].y * image.shape[0])
    return (xl,yl),(xr,yr)

def expand_point(point_a,point_b,scale = 0.1):
    # point_a =  (xl, yl)
    # point_b = (xr, yr)
    ab_distance = np.sqrt((point_b[0] - point_a[0])**2 + (point_b[1] - point_a[1])**2)
    bc_distance = ab_distance * scale
    unit_vector_ab = ((point_b[0] - point_a[0]) / ab_distance, (point_b[1] - point_a[1]) / ab_distance)
    point_r = (int(point_b[0] + bc_distance * unit_vector_ab[0]), int(point_b[1] + bc_distance * unit_vector_ab[1]))
    cv2.circle(image,point_r, 5, (0, 0, 255), -1)
    # cv2.putText(image, f"{point_r[0]}:{point_r[1]}", point_r, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    tpm = point_a
    point_a =  point_b
    point_b = tpm
    ab_distance = np.sqrt((point_b[0] - point_a[0])**2 + (point_b[1] - point_a[1])**2)
    bc_distance = ab_distance * scale
    unit_vector_ab = ((point_b[0] - point_a[0]) / ab_distance, (point_b[1] - point_a[1]) / ab_distance)
    point_l = (int(point_b[0] + bc_distance * unit_vector_ab[0]), int(point_b[1] + bc_distance * unit_vector_ab[1]))
    cv2.circle(image,point_l, 5, (0, 0, 255), -1)
    # cv2.putText(image, f"{point_l[0]}:{point_l[1]}", point_l, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    return point_l, point_r

def draw_positon(name):
    x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['LEFT_'+name]].x * image.shape[1])
    y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['LEFT_'+name]].y * image.shape[0])
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    cv2.putText(image, f"{x}:{y}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['RIGHT_'+name]].x * image.shape[1])
    y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['RIGHT_'+name]].y * image.shape[0])
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    cv2.putText(image, f"{x}:{y}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

def remove_background(bg_image):
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

    frame = cv2.flip(image, 1)
    height , width, channel = frame.shape

    frame_with_alpha = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # get the result
    results = selfie_segmentation.process(RGB)

    # extract segmented mask
    mask = results.segmentation_mask
    condition = np.stack(
        (results.segmentation_mask,) * 4, axis=-1) > 0.5

    # resize the background image to the same size of the original frame
    bg_image = cv2.resize(bg_image, (width, height))

    # combine frame and background image using the condition
    output_image = np.where(condition, frame_with_alpha, bg_image)

    return output_image

def crop_image(image,listPoint):
    pts = np.array(listPoint)
    # Find the bounding rectangle that encloses the points
    x, y, w, h = cv2.boundingRect(pts)
    image = cv2.flip(image, 1)

    # Crop the image
    crop = image[y:y+h, x:x+w]
    return crop

def border_detect(img):

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Use Canny edge detection to detect edges
    edges = cv2.Canny(blurred, 50, 150)

    # Use Hough line transformation to detect the longest line
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    # Draw the longest line on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the result
    cv2.imshow('Result', img)

def find_diagonal_intersection(p1, p2, p3, p4):
    print(p1,p2,p4,p3)
    # Find intersection of two diagonals
    y1, x1 = p1
    y2, x2 = p2
    y3, x3 = p3
    y4, x4 = p4
    
    a = y1-y2
    b = x2-x1
    c = y3-y4
    d = x4-x3
    y = (c*x3+d*y3-c*(x1+b/a*y1))/(d-c*b/a)
    x = x1+b/a*y1-b/a*y


    
    return (int(y),int(x))



# ================================================================= BODY IMPLEMENTATION


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# initialize mediapipe
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Create a background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
global results
# Load the input image
img = cv2.imread('input.png')
bg_image = cv2.imread('./transparent.png',cv2.IMREAD_UNCHANGED)

image = img
results = pose.process(image)
#  Prepare list of 4 point for crop images
listPoint = []
if results.pose_landmarks:
    shoulder_l,shoulder_r =  get_positon('SHOULDER')
    expand_point(shoulder_l,shoulder_r,scale=0.15)
    img = cv2.line(image,  shoulder_l, shoulder_r, (0, 255, 0), 1)
    hip_l, hip_r =  get_positon('HIP')
    expand_point(hip_l,hip_r,scale=0.4)
    img = cv2.line(image,  hip_l, hip_r, (0, 255, 0), 1)
    draw_positon('SHOULDER')
    draw_positon('HIP')
    draw_positon('EYE')
    draw_positon('ELBOW')
    draw_positon('WRIST')

    listPoint = [shoulder_l,hip_r,hip_l,shoulder_r]
    print(listPoint)

    intersection_point =   find_diagonal_intersection(*listPoint)
    print(intersection_point)
    cv2.circle(img, intersection_point, 5, (0, 255, 0), -1)

remove_bg_img = remove_background(bg_image)
cv2.imwrite('output_image.png', img)
# Crop image
# crop = crop_image(remove_bg_img,listPoint)
# cv2.imwrite('output_croped.png', crop)
cv2.imshow("Res",img)
# cv2.waitKey(0)
cv2.destroyAllWindows()
