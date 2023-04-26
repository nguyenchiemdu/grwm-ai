from cmath import sqrt
import cv2

from lib.get_position import get_position
from lib.draw_position import draw_position
from lib.expand_point import get_expand_point
from lib.remove_background import remove_background
from lib.crop_image import crop_image
from lib.algebra import (
    breadth_point,
    find_diagonal_intersection,
    linear_equation,
    find_slope,
    point_to_left,
    point_up,
)
from lib.pose import pose

ouput_path = "output/"
input_path = "input/"
# Load the input image
image = cv2.imread(f"{input_path}input.png")
bg_image = cv2.imread(f"{input_path}transparent.png", cv2.IMREAD_UNCHANGED)
remove_bg_img = remove_background(bg_image, image)
#  body landmarks
results = pose.process(image)
if results.pose_landmarks:
    shoulder_l, shoulder_r = get_position("SHOULDER", results, remove_bg_img)
    expanded_shoulder_l, expanded_shoulder_r = get_expand_point(shoulder_l, shoulder_r, remove_bg_img, scale=0.13)
    cv2.circle(remove_bg_img,expanded_shoulder_l,5,(0, 0, 255, 255), -1)
    cv2.circle(remove_bg_img,expanded_shoulder_r,5,(0, 0, 255, 255), -1)
    cv2.line(remove_bg_img, shoulder_l, shoulder_r, (0, 255, 0, 255), 1)

    hip_l, hip_r = get_position("HIP", results, remove_bg_img)
    expanded_hip_l, expanded_hip_r = get_expand_point(hip_l, hip_r, remove_bg_img, scale=0.4)
    cv2.circle(remove_bg_img,expanded_hip_l,5,(0, 0, 255, 255), -1)
    cv2.circle(remove_bg_img,expanded_hip_r,5,(0, 0, 255, 255), -1)
    cv2.line(remove_bg_img, hip_l, hip_r, (0, 255, 0, 255), 1)


    draw_position("SHOULDER", results, remove_bg_img)
    draw_position("HIP", results, remove_bg_img)
    draw_position("EYE", results, remove_bg_img)
    draw_position("ELBOW", results, remove_bg_img)
    draw_position("WRIST", results, remove_bg_img)


    listPoint = [shoulder_l, hip_r, hip_l, shoulder_r]
    intersection_point = find_diagonal_intersection(*listPoint)
    cv2.circle(remove_bg_img, intersection_point, 5, (0, 255, 0, 255), -1)




# # Crop image
# crop_points = [expanded_shoulder_l, expanded_shoulder_r, expanded_hip_l, expanded_hip_r]
# croped_image = crop_image(remove_bg_img, crop_points)



shoulder_slope = find_slope(shoulder_l, shoulder_r)
hip_slope = find_slope(hip_l, hip_r)
slope = (shoulder_slope + hip_slope) / 2
left,right = breadth_point(slope,intersection_point,remove_bg_img)



hip_left,hip_right = breadth_point(hip_slope,hip_l,remove_bg_img)

shoulder_left,shoulder_right = breadth_point(shoulder_slope,shoulder_l,remove_bg_img)


cv2.line(remove_bg_img, left, right, (0, 0, 255,255), 1)
cv2.circle(remove_bg_img, left, 5, (0, 0, 255, 255), -1)
cv2.circle(remove_bg_img, right, 5, (0, 0, 255, 255), -1)

cv2.line(remove_bg_img, hip_left, hip_right, (0, 255, 255,255), 1)
cv2.circle(remove_bg_img, hip_left, 5, (0, 255, 255, 255), -1)
cv2.circle(remove_bg_img, hip_right, 5, (0, 255, 255, 255), -1)

cv2.circle(remove_bg_img, shoulder_left, 5, (0, 255, 255, 255), -1)
cv2.circle(remove_bg_img, shoulder_right, 5, (0, 255, 255, 255), -1)



cv2.imwrite(f"{ouput_path}remove_bg_img.png", remove_bg_img)
# cv2.imwrite(f"{ouput_path}croped_image.png", croped_image)