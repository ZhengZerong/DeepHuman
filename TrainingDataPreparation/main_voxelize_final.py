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

from __future__ import print_function, absolute_import, division
import numpy as np
from scipy import ndimage
import scipy.io as sio
import os
from glob import glob

import config as conf
import util
from ObjIO import load_obj_data_binary
import multiprocessing


log = util.logger.write
std_faces = np.int32(np.loadtxt('faces.txt') - 1)


def binary_fill_from_corner_3D(input, structure=None, output=None, origin=0):
    mask = np.logical_not(input)
    tmp = np.zeros(mask.shape, bool)
    for xi in [0, tmp.shape[0] - 1]:
        for yi in [0, tmp.shape[1] - 1]:
            for zi in [0, tmp.shape[2] - 1]:
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


def voxelize(fname, dim_h, dim_w, binvox_dir, corner_res):
    """voxelizes a mesh (solid voxelization)"""
    volume = util.voxelize(fname, dim_h, dim_w, binvox_dir)
    volume[:corner_res, :corner_res, :corner_res] = 0
    volume[-corner_res:, -corner_res:, -corner_res:] = 0
    # volume = ndimage.binary_fill_holes(volume)
    volume = binary_fill_from_corner_3D(volume)
    return volume


def process_one_item(item, voxel_dir, dim_h, dim_w, binvox_dir, corner_res, voxel_size):
    print('Processing: ' + item)
    dir_name, fname = os.path.split(item)

    # check existence
    mat_fname = os.path.join(voxel_dir, 'voxel' + fname[4:-4] + '.mat')
    if os.path.exists(mat_fname):
        try:
            sio.loadmat(mat_fname)
        except IOError:
            print('Found corrupted data: ' + item)
        except ValueError:
            print('Found corrupted data: ' + item)
        else:
            return

    # voxelize
    mesh_file = item
    smpl_file = dir_name + '/smpl' + fname[4:]  # the corresponding SMPL file
    mesh_volume = voxelize(mesh_file, dim_h, dim_w, binvox_dir, corner_res)
    smpl_volume = voxelize(smpl_file, dim_h, dim_w, binvox_dir, corner_res)

    # calculate semantic volume
    smpl = load_obj_data_binary(smpl_file)
    smpl_v = np.copy(smpl['v'][:6890, :])
    for f, f_ in zip(smpl['f'], std_faces):
        smpl_v[f_[0]] = smpl['v'][f[0]]
        smpl_v[f_[1]] = smpl['v'][f[1]]
        smpl_v[f_[2]] = smpl['v'][f[2]]

    smpl_v_volume = util.calc_vmap_volume_fast(smpl_volume, smpl_v,
                                               dim_h, dim_w, voxel_size)
    sio.savemat(mat_fname,
                {'mesh_volume': mesh_volume, 'smpl_v_volume': smpl_v_volume},
                do_compression=True)


def main():
    output_dir = conf.output_dir
    mesh_dir = os.path.join(output_dir, './mesh_smpl/')
    voxel_dir = os.path.join(output_dir, './voxel2/')
    binvox_dir = conf.binvox_dir
    dim_h = conf.volume_h
    dim_w = conf.volume_w
    voxel_size = conf.voxel_size
    corner_res = conf.corner_res

    all_file = sorted(glob(os.path.join(mesh_dir, 'mesh*.obj')))
    log('Found %d .obj files. ' % len(all_file))

    # [single thread]
    # for f in all_file:
    #     process_one_item(f, voxel_dir, dim_h, dim_w, binvox_dir, corner_res, voxel_size)

    # [multi thread]
    pool = multiprocessing.Pool(processes=7)
    try:
        r = [pool.apply_async(process_one_item, args=(
        f, voxel_dir, dim_h, dim_w, binvox_dir, corner_res, voxel_size))
             for f in all_file]
        pool.close()
        for item in r:
            item.wait(timeout=99999999)
    except KeyboardInterrupt:
        pool.terminate()
    finally:
        pool.join()


if __name__ == '__main__':
    main()

