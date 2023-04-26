import cv2
import numpy as np

def get_expand_point(point_a,point_b,image,scale = 0.1):
    # point_a =  (xl, yl)
    # point_b = (xr, yr)
    ab_distance = np.sqrt((point_b[0] - point_a[0])**2 + (point_b[1] - point_a[1])**2)
    bc_distance = ab_distance * scale
    unit_vector_ab = ((point_b[0] - point_a[0]) / ab_distance, (point_b[1] - point_a[1]) / ab_distance)
    point_l = (int(point_b[0] + bc_distance * unit_vector_ab[0]), int(point_b[1] + bc_distance * unit_vector_ab[1]))

    tpm = point_a
    point_a =  point_b
    point_b = tpm
    ab_distance = np.sqrt((point_b[0] - point_a[0])**2 + (point_b[1] - point_a[1])**2)
    bc_distance = ab_distance * scale
    unit_vector_ab = ((point_b[0] - point_a[0]) / ab_distance, (point_b[1] - point_a[1]) / ab_distance)
    point_r = (int(point_b[0] + bc_distance * unit_vector_ab[0]), int(point_b[1] + bc_distance * unit_vector_ab[1]))

    return point_l, point_r