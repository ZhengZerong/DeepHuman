from __future__ import print_function, division
import os
import numpy as np
import math
import scipy
import scipy.io as sio
import binvox_rw
from subprocess import call
from scipy import ndimage


from TriBoxTest import tri_box_overlap


class SmplVtx(object):
    """
    Local class used to load and store SMPL's vertices coordinate at rest pose at mean shape
    """
    def __init__(self):
        self.smpl_vtx_std = np.loadtxt('./SmplUtil/vertices.txt')
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


def get_smpl_std_vertex():
    return _smpl_vtx.smpl_vtx_std


def voxelize(mesh_v, mesh_f, dim_h, dim_w, voxel_size):
    box_half_size = np.array([voxel_size/2.0, voxel_size/2.0, voxel_size/2.0])
    dim_x, dim_y, dim_z = dim_w, dim_h, dim_w
    dim_x_half, dim_y_half, dim_z_half = dim_x / 2, dim_y / 2, dim_z / 2
    new_volume = np.zeros((dim_x, dim_y, dim_z), dtype=np.uint8)

    for f in mesh_f:
        v0 = mesh_v[f[0], :]
        v1 = mesh_v[f[1], :]
        v2 = mesh_v[f[2], :]

        vs = np.asarray([v0, v1, v2])
        min_corner = np.min(vs, axis=0)
        max_corner = np.max(vs, axis=0)

        min_corner[0] = math.floor(min_corner[0] / voxel_size) + dim_x_half
        min_corner[1] = math.floor(min_corner[1] / voxel_size) + dim_y_half
        min_corner[2] = math.floor(min_corner[2] / voxel_size) + dim_z_half
        min_corner = np.int32(np.maximum(min_corner, 0))

        max_corner[0] = math.ceil(max_corner[0] / voxel_size) + dim_x_half
        max_corner[1] = math.ceil(max_corner[1] / voxel_size) + dim_y_half
        max_corner[2] = math.ceil(max_corner[2] / voxel_size) + dim_z_half
        max_corner = np.int32(np.minimum(max_corner, np.array([dim_x, dim_y, dim_z])))

        for xx in range(min_corner[0], max_corner[0]):
            for yy in range(min_corner[1], max_corner[1]):
                for zz in range(min_corner[2], max_corner[2]):
                    vxx = (xx - dim_x_half + 0.5) * voxel_size
                    vyy = (yy - dim_y_half + 0.5) * voxel_size
                    vzz = (zz - dim_z_half + 0.5) * voxel_size
                    box_center = np.array([vxx, vyy, vzz])
                    if tri_box_overlap(box_center, box_half_size, vs):
                        new_volume[xx, yy, zz] = 1

    return new_volume


def voxelize_2(mesh_path, dim_h, dim_w, voxelizer_exe_path):
    dim_x, dim_y, dim_z = dim_w, dim_h, dim_w
    new_volume = np.zeros((dim_x, dim_y, dim_z), dtype=np.uint8)

    call([os.path.join(voxelizer_exe_path, 'voxelizer'), mesh_path, mesh_path+'.occvox'])
    with open(mesh_path+'.occvox', 'r') as fp:
        for line in fp.readlines():
            indices = line.split(' ')
            vx, vy, vz = int(indices[0]), int(indices[1]), int(indices[2])
            new_volume[vx, vy, vz] = 1
    call(['rm', mesh_path+'.occvox'])
    return new_volume


def calc_vmap_volume(smpl_volume, smpl_v, dim_h, dim_w, voxel_size):
    dim_h_half = dim_h/2
    dim_w_half = dim_w/2
    sigma = 0.05*0.05
    K = 4

    smpl_std_v = get_smpl_std_vertex()

    x_dim, y_dim, z_dim = smpl_volume.shape[0], smpl_volume.shape[1], smpl_volume.shape[2]
    smpl_v_volume = np.zeros((x_dim, y_dim, z_dim, 3), dtype=np.float32)

    kd_tree = scipy.spatial.KDTree(smpl_v)
    xx, yy, zz = np.where(smpl_volume>0)
    oc_num = xx.shape[0]

    xx1 = np.expand_dims((xx - dim_w_half + 0.5) * voxel_size, axis=-1)
    yy1 = np.expand_dims((yy - dim_h_half + 0.5) * voxel_size, axis=-1)
    zz1 = np.expand_dims((zz - dim_w_half + 0.5) * voxel_size, axis=-1)

    pts = np.concatenate((xx1, yy1, zz1), axis=-1)
    dist_list, id_list = kd_tree.query(pts, k=K)

    weight_list = np.exp(-np.square(dist_list)/sigma)
    vmap = np.zeros((oc_num, 3))
    vmap_weight = np.zeros((oc_num, 1))

    for ni in range(K):
        vmap_weight[:, 0] += weight_list[:, ni]
        vmap += weight_list[:, ni:(ni+1)] * smpl_std_v[id_list[:, ni], :]

    vmap /= vmap_weight
    smpl_v_volume[xx, yy, zz, :] = vmap[:, :]

    # for xx in range(x_dim):
    #     for yy in range(y_dim):
    #         for zz in range(z_dim):
    #             if smpl_volume[xx, yy, zz] > 0:
    #                 pt = np.array([(xx - dim_w_half + 0.5) * voxel_size,
    #                                (yy - dim_h_half + 0.5) * voxel_size,
    #                                (zz - dim_w_half + 0.5) * voxel_size])
    #                 dist_list, idx_list = kd_tree.query(pt, k=4)
    #
    #                 sum_weight = 0
    #                 v_map = np.zeros((3,))
    #                 for d, i in zip(dist_list, idx_list):
    #                     w = math.exp(-d*d/sigma)
    #                     sum_weight += w
    #                     v_map += w * smpl_std_v[i, :]
    #                 v_map /= sum_weight
    #                 smpl_v_volume[xx, yy, zz, :] = v_map

    return smpl_v_volume


def calc_vmap_volume_v2(smpl_volume, smpl_v, dim_h, dim_w, voxel_size):
    dim_h_half = dim_h/2
    dim_w_half = dim_w/2
    sigma = 0.05*0.05

    smpl_std_v = get_smpl_std_vertex()

    x_dim, y_dim, z_dim = smpl_volume.shape[0], smpl_volume.shape[1], smpl_volume.shape[2]
    smpl_v_volume = np.zeros((x_dim, y_dim, z_dim, 3), dtype=np.float32)
    kd_tree = scipy.spatial.KDTree(smpl_v)

    xx_nz, yy_nz, zz_nz = smpl_volume.nonzero()
    for xx in xx_nz:
        for yy in yy_nz:
            for zz in zz_nz:
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


def load_binvox_as_volume(binvox_path, voxel_size):
    with open(binvox_path, 'rb') as f:
        binvox = binvox_rw.read_as_3d_array(f, fix_coords=True).data
    return binvox


def load_binvox_as_coords(binvox_path, voxel_size):
    with open(binvox_path, 'rb') as f:
        binvox = binvox_rw.read_as_3d_array(f, fix_coords=True).data

    x_dim, y_dim, z_dim = binvox.shape[0], binvox.shape[1], binvox.shape[2]
    dim_x_half, dim_y_half, dim_z_half = x_dim / 2, y_dim / 2, z_dim / 2
    coords_list = []
    for xx in range(x_dim):
        for yy in range(y_dim):
            for zz in range(z_dim):
                if binvox[xx, yy, zz] > 0:
                    coords_list.append([(xx - dim_x_half + 0.5) * voxel_size,
                                       (yy - dim_y_half + 0.5) * voxel_size,
                                       (zz - dim_z_half + 0.5) * voxel_size])

    coords_list = np.asarray(coords_list)
    return coords_list


def resize_volume(volume, dim_x, dim_y, dim_z):
    new_volume = np.zeros((dim_x, dim_y, dim_z), dtype=np.uint8)
    # scale_x = dim_x/volume.shape[0]
    # scale_y = dim_y/volume.shape[1]
    # scale_z = dim_z/volume.shape[2]
    #
    # for xx in range(volume.shape[0]):
    #     for yy in range(volume.shape[1]):
    #         for zz in range(volume.shape[2]):
    #             if volume[xx, yy, zz] > 0:
    #                 xxx = int(round(xx * scale_x))
    #                 yyy = int(round(yy * scale_y))
    #                 zzz = int(round(zz * scale_z))
    #                 new_volume[xxx, yyy, zzz] = 1
    scale_x = volume.shape[0]/dim_x
    scale_y = volume.shape[1]/dim_y
    scale_z = volume.shape[2]/dim_z

    print(scale_x, scale_y, scale_z)

    for xx in range(dim_x):
        for yy in range(dim_y):
            for zz in range(dim_z):
                xxx = int(round((xx+0.5) * scale_x - 0.5))
                yyy = int(round((yy+0.5) * scale_y - 0.5))
                zzz = int(round((zz+0.5) * scale_z - 0.5))
                new_volume[xx, yy, zz] = volume[xxx, yyy, zzz]

    return new_volume


def get_volume_from_points(points, dim_x, dim_y, dim_z, voxel_size):
    dim_x_half, dim_y_half, dim_z_half = dim_x / 2, dim_y / 2, dim_z / 2
    new_volume = np.zeros((dim_x, dim_y, dim_z), dtype=np.uint8)
    for p in points:
        xx = int(round(p[0]/voxel_size - 0.5 + dim_x_half))
        yy = int(round(p[1]/voxel_size - 0.5 + dim_y_half))
        zz = int(round(p[2]/voxel_size - 0.5 + dim_z_half))
        xx = min(max(0, xx), dim_x-1)
        yy = min(max(0, yy), dim_y-1)
        zz = min(max(0, zz), dim_z-1)
        new_volume[xx, yy, zz] = 1
    return new_volume


def save_volume(volume, fname, dim_h, dim_w, voxel_size):
    dim_h_half = dim_h / 2
    dim_w_half = dim_w / 2
    sigma = 0.05 * 0.05

    x_dim, y_dim, z_dim = volume.shape[0], volume.shape[1], volume.shape[2]
    with open(fname, 'wb') as fp:
        for xx in range(x_dim):
            for yy in range(y_dim):
                for zz in range(z_dim):
                    if volume[xx, yy, zz] > 0:
                        pt = np.array([(xx - dim_w_half + 0.5) * voxel_size,
                                       (yy - dim_h_half + 0.5) * voxel_size,
                                       (zz - dim_w_half + 0.5) * voxel_size])
                        fp.write('v %f %f %f\n' % (pt[0], pt[1], pt[2]))


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
                                 (pt[0], pt[1], pt[2], v_volume[xx, yy, zz, 0], v_volume[xx, yy, zz, 1], v_volume[xx, yy, zz, 2]))


def save_volume_soft(volume, fname, dim_h, dim_w, voxel_size, thres):
    dim_h_half = dim_h / 2
    dim_w_half = dim_w / 2
    sigma = 0.05 * 0.05

    x_dim, y_dim, z_dim = volume.shape[0], volume.shape[1], volume.shape[2]
    with open(fname, 'wb') as fp:
        for xx in range(x_dim):
            for yy in range(y_dim):
                for zz in range(z_dim):
                    if volume[xx, yy, zz] > thres:
                        pt = np.array([(xx - dim_w_half + 0.5) * voxel_size,
                                       (yy - dim_h_half + 0.5) * voxel_size,
                                       (zz - dim_w_half + 0.5) * voxel_size])
                        fp.write('v %f %f %f\n' % (pt[0], pt[1], pt[2]))


def load_volume_from_mat(fname):
    return sio.loadmat(fname)


def rotate_volume(volume, view_id):
    new_volume = volume
    if view_id == 1:    # z-->x, (-x)-->z
        if len(new_volume.shape) == 3:
            new_volume = np.transpose(new_volume, (2, 1, 0))
        elif len(new_volume.shape) == 4:
            new_volume = np.transpose(new_volume, (2, 1, 0, 3))
        new_volume = np.flip(new_volume, axis=2)
    elif view_id == 2:
        new_volume = np.flip(new_volume, axis=0)
        new_volume = np.flip(new_volume, axis=2)
    elif view_id == 3:
        if len(new_volume.shape) == 3:
            new_volume = np.transpose(new_volume, (2, 1, 0))
        elif len(new_volume.shape) == 4:
            new_volume = np.transpose(new_volume, (2, 1, 0, 3))
        new_volume = np.flip(new_volume, axis=0)
    return new_volume


def binary_fill_from_corner_3D(input, structure=None, output=None, origin=0):
    mask = np.logical_not(input)
    tmp = np.zeros(mask.shape, bool)
    for xi in [0, tmp.shape[0]-1]:
        for yi in [0, tmp.shape[1]-1]:
            for zi in [0, tmp.shape[2]-1]:
                tmp[xi, yi, zi] = True
    inplace = isinstance(output, np.ndarray)
    if inplace:
        ndimage.binary_dilation(tmp, structure=structure, iterations=-1,
                                mask=mask, output=output, border_value=0,
                                origin=origin)
        np.logical_not(output, output)
    else:
        output = ndimage.binary_dilation(tmp, structure=structure, iterations=-1,
                                         mask=mask, border_value=0,
                                         origin=origin)
        np.logical_not(output, output)
        return output

