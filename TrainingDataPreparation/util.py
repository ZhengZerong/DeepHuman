# Software License Agreement (BSD License)
#
# Copyright (c) 2019, Zerong Zheng (zzr18@mails.tsinghua.edu.cn)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the <organization> nor the
#  names of its contributors may be used to endorse or promote products
#  derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Utilization"""

from __future__ import print_function, absolute_import, division
import numpy as np
import scipy
import math
import os
import sys
import cv2 as cv
import datetime
from subprocess import call

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
    """performs mkdir after checking existence"""
    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        logger.write('WARNING: %s already exists. ' % dir)


class ProgressBar(object):
    def __init__(self, width=40):
        self._w = width
        self._total = 1
        self._curr = 0
        self._curr_str = ''

    def start(self, total_count):
        self._total = total_count
        self._curr = 0
        self._curr_str = "[%s%s] (%d/%d)" % ('', ' '*self._w,
                                             self._curr, self._total)
        sys.stdout.write(self._curr_str)
        sys.stdout.flush()

    def count(self, c=1):
        # remove previous output
        sys.stdout.write("\b" * len(self._curr_str))
        sys.stdout.flush()

        # update output
        self._curr = self._curr + c
        step = int(self._w * self._curr/self._total)
        self._curr_str = "[%s%s] (%d/%d)" % ('#'*step, ' '*(self._w-step),
                                             self._curr, self._total)
        sys.stdout.write(self._curr_str)
        sys.stdout.flush()

    def end(self):
        sys.stdout.write('\nFinished. \n')
        sys.stdout.flush()
        self._total = 1
        self._curr = 0
        self._curr_str = ''


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


# parameters sampling
# =====================================================
def sample_bg_img(bg_list, bg_root, w=256, h=256):
    """samples a background image and pre-processes it"""
    n = len(bg_list)
    i = np.random.randint(0, n, 1, dtype=np.int32)
    img_dir = os.path.join(bg_root, bg_list[i[0]])
    img = cv.imread(img_dir)
    if img is None:
        return np.ones((h, w, 3), np.float32), 'none'

    # convert color
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv.cvtColor(img, cv.COLOR_BGRA2RGB)

    # crop
    w_, h_ = img.shape[1], img.shape[0]
    img = img[(h_//2-128):(h_//2+128), (w_//2-128):(w_//2+128), :]/255.0

    if not w == 256 or not h == 256:
        img = cv.resize(img, (w, h))

    return img, bg_list[i[0]]


def sample_sh_component():
    """samples Spherical Harmonics components"""
    sh = np.array([3.5, 0, 0., 0., 0., 0., 0., 0., 0.])
    sh += np.random.rand(9)*1.4 - 0.7  # sample between [-0.7, 0.7]
    return sh


def sample_verticle_lighting(light_num=3):
    """samples vertical lighting position and power (color)"""
    light_pos = np.zeros((light_num, 3))
    light_pos[:, 1] = 10    # prefer vertical lighting
    light_pos += np.random.randn(light_num, 3) * 2   # randomly move lighting source
    light_color = np.ones(3) * 0.75
    light_color += np.random.rand() * 0.5 - 0.25
    return light_pos, light_color


def calc_transform_params(mesh, smpl, hb_ratio=1.0, scale_noise=0):
    """
    Calculates the transformation params used to transform the mesh to unit
    bounding box centered at the origin. Returns translation and scale.
    Note that to use the returned parameters, you should perform translation
    first, followed by scaling
    """
    min_x = np.min(mesh['v'][:, 0])
    max_x = np.max(mesh['v'][:, 0])
    min_y = np.min(mesh['v'][:, 1])
    max_y = np.max(mesh['v'][:, 1])
    min_z = np.min(mesh['v'][:, 2])
    max_z = np.max(mesh['v'][:, 2])

    min_x = min(np.min(smpl['v'][:, 0]), min_x)
    max_x = max(np.max(smpl['v'][:, 0]), max_x)
    min_y = min(np.min(smpl['v'][:, 1]), min_y)
    max_y = max(np.max(smpl['v'][:, 1]), max_y)
    min_z = min(np.min(smpl['v'][:, 2]), min_z)
    max_z = max(np.max(smpl['v'][:, 2]), max_z)

    trans = -np.array([(min_x + max_x) / 2, (min_y + max_y) / 2,
                       (min_z + max_z) / 2])

    scale_inv = max(max((max_x-min_x)/hb_ratio, (max_y-min_y)),
                    (max_z-min_z)/hb_ratio)
    scale_inv *= (1.05 + scale_noise)
    scale_inv += 1e-3   # avoid division by zero
    scale = 1.0 / scale_inv
    return trans, scale


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


# voxelization
# =====================================================
def voxelize(mesh_path, dim_h, dim_w, voxelizer_exe_path):
    """voxelizes the mesh using the given binary executable program"""
    call([os.path.join(voxelizer_exe_path, 'voxelizer'), mesh_path,
          mesh_path+'.occvox'])

    dim_x, dim_y, dim_z = dim_w, dim_h, dim_w
    new_volume = np.zeros((dim_x, dim_y, dim_z), dtype=np.uint8)
    with open(mesh_path+'.occvox', 'r') as fp:
        for line in fp.readlines():
            indices = line.split(' ')
            vx, vy, vz = int(indices[0]), int(indices[1]), int(indices[2])
            new_volume[vx, vy, vz] = 1
    call(['rm', mesh_path+'.occvox'])
    return new_volume


def calc_vmap_volume(smpl_volume, smpl_v, dim_h, dim_w, voxel_size):
    """calculates the semantic volume"""
    dim_h_half = dim_h/2
    dim_w_half = dim_w/2
    sigma = 0.05*0.05

    smpl_std_v = get_smpl_semantic_code()

    x_dim = smpl_volume.shape[0]
    y_dim = smpl_volume.shape[1]
    z_dim = smpl_volume.shape[2]
    smpl_v_volume = np.zeros((x_dim, y_dim, z_dim, 3), dtype=np.float32)

    # for each solid voxel, searches its KNN on SMPL model, and averages
    # the neighbors' semantic code to obtain the voxel's semantic code
    kd_tree = scipy.spatial.KDTree(smpl_v)
    for xx in range(x_dim):
        for yy in range(y_dim):
            for zz in range(z_dim):
                if smpl_volume[xx, yy, zz] > 0:
                    pt = np.array([(xx - dim_w_half + 0.5) * voxel_size,
                                   (yy - dim_h_half + 0.5) * voxel_size,
                                   (zz - dim_w_half + 0.5) * voxel_size])
                    dist_list, idx_list = kd_tree.query(pt, k=4)

                    sum_weight = 0
                    v_map = np.zeros((3,))
                    for d, i in zip(dist_list, idx_list):
                        w = math.exp(-d*d/sigma)
                        sum_weight += w
                        v_map += w * smpl_std_v[i, :]
                    v_map /= sum_weight
                    smpl_v_volume[xx, yy, zz, :] = v_map

    return smpl_v_volume


def calc_vmap_volume_fast(smpl_volume, smpl_v, dim_h, dim_w, voxel_size):
    """calculates the semantic volume without for-loop"""
    dim_h_half = dim_h/2
    dim_w_half = dim_w/2
    sigma = 0.05*0.05

    smpl_std_v = get_smpl_semantic_code()

    x_dim = smpl_volume.shape[0]
    y_dim = smpl_volume.shape[1]
    z_dim = smpl_volume.shape[2]
    smpl_v_volume = np.zeros((x_dim, y_dim, z_dim, 3), dtype=np.float32)

    kd_tree = scipy.spatial.KDTree(smpl_v)

    # gets the solid voxels
    pt_entry = np.where(smpl_volume > 0)
    pt_num = pt_entry[0].shape[0]
    pt_coord = np.zeros((pt_num, 3), dtype=np.float32)
    pt_coord[:, 0] = (pt_entry[0] - dim_w_half + 0.5) * voxel_size
    pt_coord[:, 1] = (pt_entry[1] - dim_h_half + 0.5) * voxel_size
    pt_coord[:, 2] = (pt_entry[2] - dim_w_half + 0.5) * voxel_size

    # for each solid voxel, searches its KNN on SMPL model, and averages
    # the neighbors' semantic code to obtain the voxel's semantic code
    dist_list, idx_list = kd_tree.query(pt_coord, k=4)  # both variables have shape (k, 4)
    w = np.exp(-dist_list*dist_list/sigma)
    sum_weight = np.sum(w, axis=1, keepdims=True)
    w /= sum_weight
    v_map = np.zeros((pt_num, 3))
    v_map += np.reshape(w[:, 0], (-1, 1)) * smpl_std_v[idx_list[:, 0], :]
    v_map += np.reshape(w[:, 1], (-1, 1)) * smpl_std_v[idx_list[:, 1], :]
    v_map += np.reshape(w[:, 2], (-1, 1)) * smpl_std_v[idx_list[:, 2], :]
    v_map += np.reshape(w[:, 3], (-1, 1)) * smpl_std_v[idx_list[:, 3], :]

    # restores the semantic code list into a volume
    channel_entry = ((np.zeros((pt_num, ), dtype=np.int32), ),
                     (np.ones((pt_num, ), dtype=np.int32), ),
                     (2 * np.ones((pt_num, ), dtype=np.int32), ))
    smpl_v_volume[pt_entry + channel_entry[0]] = v_map[:, 0]
    smpl_v_volume[pt_entry + channel_entry[1]] = v_map[:, 1]
    smpl_v_volume[pt_entry + channel_entry[2]] = v_map[:, 2]

    return smpl_v_volume


def save_v_volume(v_volume, fname, dim_h, dim_w, voxel_size):
    dim_h_half = dim_h / 2
    dim_w_half = dim_w / 2
    sigma = 0.05 * 0.05

    x_dim, y_dim, z_dim = v_volume.shape[0], v_volume.shape[1], v_volume.shape[2]
    with open(fname, 'wb') as fp:
        for xx in range(x_dim):
            for yy in range(y_dim):
                for zz in range(z_dim):
                    if (v_volume[xx, yy, zz, :] != np.zeros((3,), dtype=np.float32)).any():
                        pt = np.array([(xx - dim_w_half + 0.5) * voxel_size,
                                       (yy - dim_h_half + 0.5) * voxel_size,
                                       (zz - dim_w_half + 0.5) * voxel_size])
                        fp.write('v %f %f %f %f %f %f\n' %
                                 (pt[0], pt[1], pt[2], v_volume[xx, yy, zz, 0],
                                  v_volume[xx, yy, zz, 1], v_volume[xx, yy, zz, 2]))
