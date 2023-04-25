from cmath import sqrt
import cv2

from lib.get_position import get_position
from lib.draw_position import draw_position
from lib.expand_point import expand_point
from lib.remove_background import remove_background
from lib.crop_image import crop_image
from lib.algebra import find_diagonal_intersection,linear_equation,find_slope,point_to_left,point_to_right
from lib.pose import pose

ouput_path = "output/"
input_path = "input/"
# Load the input image
image = cv2.imread(f"{input_path}input.png")
bg_image = cv2.imread(f"{input_path}transparent.png", cv2.IMREAD_UNCHANGED)

#  body landmarks
results = pose.process(image)
#  Prepare list of 4 point for crop images
listPoint = []
if results.pose_landmarks:
    shoulder_l, shoulder_r = get_position("SHOULDER", results, image)
    topl,topr = expand_point(shoulder_l, shoulder_r, image, scale=0.15)
    cv2.line(image, shoulder_l, shoulder_r, (0, 255, 0), 1)
    hip_l, hip_r = get_position("HIP", results, image)
    botl,botr = expand_point(hip_l, hip_r, image, scale=0.4)
    cv2.line(image, hip_l, hip_r, (0, 255, 0), 1)
    draw_position("SHOULDER", results, image,put_text=True)
    draw_position("HIP", results, image)
    draw_position("EYE", results, image)
    draw_position("ELBOW", results, image)
    draw_position("WRIST", results, image)
    listPoint = [shoulder_l, hip_r, hip_l, shoulder_r]
    intersection_point = find_diagonal_intersection(*listPoint)
    cv2.circle(image, intersection_point, 5, (0, 255, 0), -1)

    


cv2.imwrite(f"{ouput_path}point_img.png", image)

remove_bg_img = remove_background(bg_image, image)
# Crop image
crop_points = [topl,topr,botl,botr]
croped_image = crop_image(remove_bg_img, crop_points)
# cv2.waitKey(0)
# Example usage

shoulder_slope  = find_slope(shoulder_l,shoulder_r)
hip_slope  = find_slope(hip_l,hip_r)
slope = (shoulder_slope+hip_slope)/2
A,B,C = linear_equation(slope,intersection_point)
point = intersection_point
# print(point[0],point[1])
# print(remove_bg_img[point[0],point[1]])
remove_bg_img  = cv2.flip(remove_bg_img,1)
while (remove_bg_img[point[1],point[0],3] >0):
    # cv2.circle(remove_bg_img, point, 5, (0,0,255,1), -1)
    point = point_to_right(A,B,C,5, point)
    # print(point)
right = point
point = intersection_point
while (remove_bg_img[point[1],point[0],3] >0):
    # cv2.circle(remove_bg_img, point, 5, (0,0,255,1), -1)
    point = point_to_left(A,B,C,5, point)
    # print(point) 
left = point

cv2.line(remove_bg_img, left, right, (0, 255, 0), 1)
cv2.circle(remove_bg_img, left, 5, (0,0,255,255),-1)
cv2.circle(remove_bg_img, right, 5, (0,0,255,255), -1)
cv2.imwrite(f"{ouput_path}remove_bg_img.png", remove_bg_img)
cv2.imwrite(f"{ouput_path}croped_image.png", croped_image)
