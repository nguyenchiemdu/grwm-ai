from cmath import sqrt


def find_diagonal_intersection(p1, p2, p3, p4):
    # Find intersection of two diagonals
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    a = y1-y2
    b = x2-x1
    c = y3-y4
    d = x4-x3
    y = (c*x3+d*y3-c*(x1+b/a*y1))/(d-c*b/a)
    x = x1+b/a*y1-b/a*y
    return (int(x),int(y))

def find_slope(point_a, point_b):
    x1,y1 = point_a
    x2,y2 = point_b
    return ((y1-y2)/(x1-x2))

def linear_equation(slope, point):
    # slope-intercept form of a line: y = mx + b
    # convert to standard form: Ax + By + C = 0
    # A = -m, B = 1, C = -(y - mx)

    x1, y1 = point  # unpack the coordinates of the point
    m = slope  # slope of the line
    A = -m
    B = 1
    C = -(y1 - m*x1)

    return A, B, C

def point_to_left(A, B, C, d, point):
    x,y = point
    slope = -A/B
    cosAlpha = sqrt(1/(1+ slope**2))
    dx = d*cosAlpha
    new_x = x - dx
    new_y = - (C +A*new_x)/B
    return int(new_x.real), int(new_y.real)

def point_to_right(A, B, C, d, point):
    x,y = point
    slope = -A/B
    cosAlpha = sqrt(1/(1+ slope**2))
    dx = d*cosAlpha
    new_x = x + dx
    new_y = - (C +A*new_x)/B
    return int(new_x.real), int(new_y.real)

    



