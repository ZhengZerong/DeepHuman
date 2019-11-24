from __future__ import division, absolute_import, print_function

"""
Triangle-Box Overlap Test
Modified from: http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox3.txt
"""

import numpy as np


X = 0
Y = 1
Z = 2


def find_min_max(x0, x1, x2):
    return min(min(x0, x1), x2), max(max(x0, x1), x2)


def plane_box_overlap(normal, vert, maxbox):
    vmin, vmax = np.zeros((3, )), np.zeros((3, ))
    for q in range(3):
        v = vert[q]
        if normal[q] > 0.:
            vmin[q] = -maxbox[q] - v
            vmax[q] = maxbox[q] - v
        else:
            vmin[q] = maxbox[q] - v
            vmax[q] = -maxbox[q] - v

    if np.dot(normal, vmin) > 0:
        return False
    if np.dot(normal, vmax) >= 0:
        return True
    return False


def AXISTEST_X01(a, b, fa, fb, v0, v1, v2, boxhalfsize):
    p0 = a*v0[Y] - b*v0[Z]
    p2 = a*v2[Y] - b*v2[Z]
    if p0 < p2:
        mn = p0
        mx = p2
    else:
        mn = p2
        mx = p0
    rad = fa * boxhalfsize[Y] + fb * boxhalfsize[Z]
    if mn>rad or mx<-rad:
        return False
    return True


def AXISTEST_X2(a, b, fa, fb, v0, v1, v2, boxhalfsize):
    p0 = a * v0[Y] - b * v0[Z]
    p1 = a * v1[Y] - b * v1[Z]
    if p0 < p1:
        mn = p0
        mx = p1
    else:
        mn = p1
        mx = p0
    rad = fa * boxhalfsize[Y] + fb * boxhalfsize[Z]
    if mn > rad or mx < -rad:
        return False
    return True

def AXISTEST_Y02(a, b, fa, fb, v0, v1, v2, boxhalfsize):
    p0 = -a * v0[X] + b * v0[Z]
    p2 = -a * v2[X] + b * v2[Z]
    if p0 < p2:
        mn = p0
        mx = p2
    else:
        mn = p2
        mx = p0
    rad = fa * boxhalfsize[X] + fb * boxhalfsize[Z]
    if mn > rad or mx < -rad:
        return False
    return True


def AXISTEST_Y1(a, b, fa, fb, v0, v1, v2, boxhalfsize):
    p0 = -a * v0[X] + b * v0[Z]
    p1 = -a * v1[X] + b * v1[Z]
    if p0 < p1:
        mn = p0
        mx = p1
    else:
        mn = p1
        mx = p0
    rad = fa * boxhalfsize[X] + fb * boxhalfsize[Z]
    if mn > rad or mx < -rad:
        return False
    return True


def AXISTEST_Z12(a, b, fa, fb, v0, v1, v2, boxhalfsize):
    p1 = a * v1[X] - b * v1[Y]
    p2 = a * v2[X] - b * v2[Y]
    if p2 < p1:
        mn = p2
        mx = p1
    else:
        mn = p1
        mx = p2
    rad = fa * boxhalfsize[X] + fb * boxhalfsize[Y]
    if mn > rad or mx < -rad:
        return False
    return True


def AXISTEST_Z0(a, b, fa, fb, v0, v1, v2, boxhalfsize):
    p0 = a * v0[X] - b * v0[Y]
    p1 = a * v1[X] - b * v1[Y]
    if p0 < p1:
        mn = p0
        mx = p1
    else:
        mn = p1
        mx = p0
    rad = fa * boxhalfsize[X] + fb * boxhalfsize[Y]
    if mn > rad or mx < -rad:
        return False
    return True


def tri_box_overlap(box_center, box_half_size, triverts):
    v0 = triverts[0, :] - box_center
    v1 = triverts[1, :] - box_center
    v2 = triverts[2, :] - box_center

    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    fex = abs(e0[X])
    fey = abs(e0[Y])
    fez = abs(e0[Z])

    if not AXISTEST_X01(e0[Z], e0[Y], fez, fey, v0, v1, v2, box_half_size): return False
    if not AXISTEST_Y02(e0[Z], e0[X], fez, fex, v0, v1, v2, box_half_size): return False
    if not AXISTEST_Z12(e0[Y], e0[X], fey, fex, v0, v1, v2, box_half_size): return False

    fex = abs(e1[X])
    fey = abs(e1[Y])
    fez = abs(e1[Z])
    if not AXISTEST_X01(e1[Z], e1[Y], fez, fey, v0, v1, v2, box_half_size): return False
    if not AXISTEST_Y02(e1[Z], e1[X], fez, fex, v0, v1, v2, box_half_size): return False
    if not AXISTEST_Z0(e1[Y], e1[X], fey, fex, v0, v1, v2, box_half_size): return False

    fex = abs(e2[X])
    fey = abs(e2[Y])
    fez = abs(e2[Z])
    if not AXISTEST_X2(e2[Z], e2[Y], fez, fey, v0, v1, v2, box_half_size): return False
    if not AXISTEST_Y1(e2[Z], e2[X], fez, fex, v0, v1, v2, box_half_size): return False
    if not AXISTEST_Z12(e2[Y], e2[X], fey, fex, v0, v1, v2, box_half_size): return False

    mn, mx = find_min_max(v0[X], v1[X], v2[X])
    if mn > box_half_size[X] or mx < -box_half_size[X]: return False

    mn, mx = find_min_max(v0[Y], v1[Y], v2[Y])
    if mn > box_half_size[Y] or mx < -box_half_size[Y]: return False

    mn, mx = find_min_max(v0[Z], v1[Z], v2[Z])
    if mn > box_half_size[Z] or mx < -box_half_size[Z]: return False

    normal = np.cross(e0, v1)
    if not plane_box_overlap(normal, v0, box_half_size): return False

    return True

# if __name__ == '__main__':
#     box_center = np.array([0.5, 0.5, 0.5])
#     box_half_size = np.array([1.5, 1.5, 1.5])
#     # triverts = np.array([[-0.2, -0.4, -0.6], [-0.4, -0.2, -0.6], [-0.1, -0.1, -0.1]])
#     triverts = np.array([[0.2, 0.4, 0.6], [0.4, 0.2, 0.6], [0.1, 0.1, 0.1]])
#     print(tri_box_overlap(box_center, box_half_size, triverts))