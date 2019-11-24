from __future__ import absolute_import, division, print_function
import datetime
import os
import numpy as np
import cv2 as cv
import datetime
from opendr.lighting import VertNormals


class Logger(object):
    def __init__(self):
        self.file = None
        self.buffer = ''

    def set_log_file(self, filename):
        assert self.file is None
        self.file = open(filename, 'wt')
        if self.buffer is not None:
            self.file.write(self.buffer)
            self.buffer = None

    def write(self, *args):
        now = datetime.datetime.now()
        dtstr = now.strftime('%Y-%m-%d %H:%M:%S')
        t_msg = '[%s]' % dtstr + ' %s' % ' '.join(map(str, args))

        print(t_msg)
        if self.file is not None:
            self.file.write(t_msg + '\n')
        else:
            self.buffer += t_msg

    def flush(self):
        if self.file is not None:
            self.file.flush()

logger = Logger()


def safe_mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


# SMPL semantic code
# =====================================================

class SmplVtx(object):
    """
    Local class used to load and store SMPL's vertices coordinate at rest pose
    with mean shape
    """
    def __init__(self):
        self.smpl_vtx_std = np.loadtxt('vertices.txt')
        min_x = np.min(self.smpl_vtx_std[:, 0])
        max_x = np.max(self.smpl_vtx_std[:, 0])
        min_y = np.min(self.smpl_vtx_std[:, 1])
        max_y = np.max(self.smpl_vtx_std[:, 1])
        min_z = np.min(self.smpl_vtx_std[:, 2])
        max_z = np.max(self.smpl_vtx_std[:, 2])

        self.smpl_vtx_std[:, 0] = (self.smpl_vtx_std[:, 0]-min_x)/(max_x-min_x)
        self.smpl_vtx_std[:, 1] = (self.smpl_vtx_std[:, 1]-min_y)/(max_y-min_y)
        self.smpl_vtx_std[:, 2] = (self.smpl_vtx_std[:, 2]-min_z)/(max_z-min_z)


_smpl_vtx = SmplVtx()


def get_smpl_semantic_code():
    """gets semantic code definition on SMPL model"""
    return _smpl_vtx.smpl_vtx_std


# mesh pre-processing
# =====================================================
def calc_normal(mesh):
    """calculates surface normal"""
    n = VertNormals(f=mesh['f'], v=mesh['v'])
    return n.r


def flip_axis_in_place(mesh, x_sign, y_sign, z_sign):
    """flips model along some axes"""
    mesh['v'][:, 0] *= x_sign
    mesh['v'][:, 1] *= y_sign
    mesh['v'][:, 2] *= z_sign

    if mesh['vn'] is not None and len(mesh['vn'].shape) == 2:
        mesh['vn'][:, 0] *= x_sign
        mesh['vn'][:, 1] *= y_sign
        mesh['vn'][:, 2] *= z_sign
    return mesh


def transform_mesh_in_place(mesh, trans, scale):
    """
    Transforms mesh
    Note that it will perform translation first, followed by scaling
    Also note that the transformation happens in-place
    """
    mesh['v'][:, 0] += trans[0]
    mesh['v'][:, 1] += trans[1]
    mesh['v'][:, 2] += trans[2]

    mesh['v'] *= scale
    return mesh


def rotate_model_in_place(mesh, x_r, y_r, z_r):
    """rotates model (x-axis first, then y-axis, and then z-axis)"""
    mat_x, _ = cv.Rodrigues(np.asarray([x_r, 0, 0], dtype=np.float32))
    mat_y, _ = cv.Rodrigues(np.asarray([0, y_r, 0], dtype=np.float32))
    mat_z, _ = cv.Rodrigues(np.asarray([0, 0, z_r], dtype=np.float32))
    mat = np.matmul(np.matmul(mat_x, mat_y), mat_z)

    v = mesh['v'].transpose()
    v = np.matmul(mat, v)
    mesh['v'] = v.transpose()

    if 'vn' in mesh and mesh['vn'] is not None and len(mesh['vn'].shape) == 2:
        n = mesh['vn'].transpose()
        n = np.matmul(mat, n)
        mesh['vn'] = n.transpose()

    return mesh


