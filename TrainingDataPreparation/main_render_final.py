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
import chumpy as ch
import os
import cv2 as cv
import json

import config as conf
import util
import renderers as rd
from ObjIO import load_obj_data, save_obj_data_binary_with_corner


log = util.logger.write


def make_output_dir(out_dir):
    """creates output folders"""
    util.safe_mkdir(out_dir)
    util.safe_mkdir(os.path.join(out_dir, 'color'))
    util.safe_mkdir(os.path.join(out_dir, 'mask'))
    util.safe_mkdir(os.path.join(out_dir, 'vmap'))
    util.safe_mkdir(os.path.join(out_dir, 'mesh_smpl'))
    util.safe_mkdir(os.path.join(out_dir, 'voxel2'))
    util.safe_mkdir(os.path.join(out_dir, 'normal'))
    util.safe_mkdir(os.path.join(out_dir, 'params'))


def load_data_list(dataset_dir):
    """loads the list of 3D textured models"""
    data_list_fname = os.path.join(dataset_dir, 'data_list.txt')
    data_list = []
    with open(data_list_fname, 'r') as fp:
        for line in fp.readlines():
            data_list.append(line[:-1])     # discard line ending symbol
    log('data list loaded. ')
    return data_list


def load_bg_list(bg_dir):
    """loads the list of background images"""
    bg_list_fname = os.path.join(bg_dir, 'img_list.txt')
    bg_list = []
    with open(bg_list_fname, 'r') as fp:
        for line in fp.readlines():
            bg_list.append(line[:-1])  # discard line ending symbol
    log('background list loaded. ')
    return bg_list


def check_rendered_img_existence(output_dir, model_idx):
    """check whether the model has been processed"""
    rendered = True
    img_idices = [4*model_idx, 4*model_idx+1, 4*model_idx+2, 4*model_idx+3]
    for ind in img_idices:
        p = dict()
        p['color'] = '%s/color/color_%08d.jpg' % (output_dir, ind)
        p['normal'] = '%s/normal/normal_%08d.png' % (output_dir, ind)
        p['mask'] = '%s/mask/mask_%08d.png' % (output_dir, ind)
        p['vmap'] = '%s/vmap/vmap_%08d.png' % (output_dir, ind)
        p['params'] = '%s/params/params_%08d.json' % (output_dir, ind)

        for pi in p.values():
            rendered = rendered and os.path.exists(pi)
    return rendered


def load_models(dataset_dir, data_item, axis_transformation):
    """loads the model and corrects the orientation"""
    mesh_dir = os.path.join(dataset_dir, data_item, 'mesh.obj')
    smpl_dir = os.path.join(dataset_dir, data_item, 'smpl.obj')
    mesh = load_obj_data(mesh_dir)
    smpl = load_obj_data(smpl_dir)
    util.flip_axis_in_place(mesh, axis_transformation[0],
                            axis_transformation[1], axis_transformation[2])
    util.flip_axis_in_place(smpl, axis_transformation[0],
                            axis_transformation[1], axis_transformation[2])
    return mesh, smpl


def save_model_for_voxelization(mesh, smpl, min_corner, max_corner,
                                output_dir, data_idx):
    """saves model to an .obj file for voxelization"""
    # create new dictionaries to remove useless info
    mesh_ = dict()
    mesh_['v'] = np.copy(mesh['v'])
    mesh_['f'] = np.copy(mesh['f'])
    smpl_ = dict()
    smpl_['v'] = np.copy(smpl['v'])
    smpl_['f'] = np.copy(smpl['f'])

    # save models for voxelization
    m_path = '%s/mesh_smpl/mesh_%08d.obj' % (output_dir, data_idx * 4)
    s_path = '%s/mesh_smpl/smpl_%08d.obj' % (output_dir, data_idx * 4)
    save_obj_data_binary_with_corner(mesh_, min_corner, max_corner,
                                     conf.corner_size, m_path)
    save_obj_data_binary_with_corner(smpl, min_corner, max_corner,
                                     conf.corner_size, s_path)


def transform_model_randomly(mesh, smpl, hb_ratio):
    """translates the model to the origin, and rotates it randomly"""
    # random rotation
    y_rot = np.random.rand(1) * np.pi * 2
    x_rot = np.random.rand(1) * 0.6 - 0.3
    z_rot = np.random.rand(1) * 0.6 - 0.3
    mesh = util.rotate_model_in_place(mesh, x_rot, y_rot, z_rot)
    smpl = util.rotate_model_in_place(smpl, x_rot, y_rot, z_rot)

    # transform the mesh and SMPL to unit bounding box
    # [-0.333, 0.333]x[-0.5, 0.5]x[-0.333, 0.333]
    s_noise = np.random.rand(1) * 0.15
    trans, scale = util.calc_transform_params(mesh, smpl, hb_ratio, s_noise)
    util.transform_mesh_in_place(mesh, trans, scale)
    util.transform_mesh_in_place(smpl, trans, scale)

    bbox_p1 = np.array([np.min(mesh['v'][:, 0]), np.min(mesh['v'][:, 1]),
                        np.min(mesh['v'][:, 2])])
    bbox_p2 = np.array([np.max(mesh['v'][:, 0]), np.max(mesh['v'][:, 1]),
                        np.max(mesh['v'][:, 2])])

    # create a dict of transformation parameters
    param = dict()
    param['trans'] = trans
    param['scale'] = scale
    param['rot'] = np.array([x_rot, y_rot, z_rot])
    param['mesh_bbox'] = np.concatenate([bbox_p1, bbox_p2])
    return mesh, smpl, param


def save_rendered_data(img, msk, nml, smap, output_dir, img_idx):
    """saves rendered images with correct format"""
    img = np.uint8(img * 255)
    msk = np.uint8(msk[:, :, 0] * 255)
    nml = np.uint16(nml * 65535)
    vmap = np.uint8(smap * 255)

    cv.imwrite('%s/color/color_%08d.jpg' % (output_dir, img_idx),
               cv.cvtColor(img, cv.COLOR_RGB2BGR))
    cv.imwrite('%s/mask/mask_%08d.png' % (output_dir, img_idx), msk)
    cv.imwrite('%s/normal/normal_%08d.png' % (output_dir, img_idx), nml)
    cv.imwrite('%s/vmap/vmap_%08d.png' % (output_dir, img_idx),
               cv.cvtColor(vmap, cv.COLOR_BGR2RGB))


def save_render_params(param, output_dir, img_idx):
    """saves rendering parameters to a .json file"""
    param_ = dict(param.items())
    for key in param_.keys():
        if isinstance(param_[key], np.ndarray):
            param_[key] = np.reshape(param_[key], (-1, )).tolist()
    s = json.dumps(param_, indent=4)
    json_path = '%s/params/params_%08d.json' % (output_dir, img_idx)
    with open(json_path, 'w') as fp:
        fp.write('// params_%08d.json\n' % img_idx)
        fp.write(s)

    # check validity of the bounding box
    if param['mesh_bbox'][0] <= -0.333 or param['mesh_bbox'][1] <= -0.5 or \
            param['mesh_bbox'][2] <= -0.333 or param['mesh_bbox'][3] >= 0.333 or \
            param['mesh_bbox'][4] >= 0.5 or param['mesh_bbox'][5] >= 0.333:
        print('Invalid mesh!! Index = %d' % img_idx)
        import pdb
        pdb.set_trace()


def main():
    output_dir = conf.output_dir
    bg_dir = conf.bg_dir
    hb_ratio = conf.hb_ratio
    dataset_dir = conf.dataset_dir
    render_img_w = conf.render_img_w
    render_img_h = conf.render_img_h

    np.random.seed()
    make_output_dir(output_dir)

    data_list = load_data_list(dataset_dir)
    bg_list = load_bg_list(bg_dir)

    # corner of unit bounding box [-0.333, 0.333]x[-0.5, 0.5]x[-0.333, 0.333]
    min_corner = np.array([-hb_ratio, -1., -hb_ratio]) * 0.5
    max_corner = -min_corner

    pb = util.ProgressBar(80)
    pb.start(len(data_list)*4)

    for di, data_item in enumerate(data_list):
        if check_rendered_img_existence(output_dir, di):
            pb.count(c=4)
            continue

        # preprocess 3D models
        mesh, smpl = load_models(dataset_dir, data_item, conf.axis_transformation)
        mesh, smpl, param_0 = transform_model_randomly(mesh, smpl, hb_ratio)
        save_model_for_voxelization(mesh, smpl, min_corner, max_corner, output_dir, di)

        # for each model, I render 4 tuples of images
        # Note that to reduce storage consumption, I use a trick; that is, I render
        # data from 4 orthogonal viewpoints (front/back/left/right), so that the
        # voxelization data in the front viewpoint can be reused in other viewpoints
        img_indices = [4 * di, 4 * di + 1, 4 * di + 2, 4 * di + 3]
        for vi in range(4):
            img_ind = img_indices[vi]
            bg, bg_fname = util.sample_bg_img(bg_list, bg_dir,
                                              render_img_w, render_img_h)
            sh = util.sample_sh_component()
            vl_pos, vl_clr = util.sample_verticle_lighting(3)
            cam_t, cam_r = ch.array((0, 0, 2.0)), ch.array((3.14, 0, 0))
            img, msk, nml, smap = rd.render_training_pairs(mesh, smpl,
                                                           render_img_w, render_img_h,
                                                           cam_r, cam_t, bg,
                                                           sh_comps=sh,
                                                           light_c=ch.ones(3),
                                                           vlight_pos=vl_pos,
                                                           vlight_color=vl_clr)
            save_rendered_data(img, msk, nml, smap, output_dir, img_ind)

            # save parameters to a json file
            param_1 = dict()
            param_1['bg_fname'] = bg_fname
            param_1['sh'] = sh
            param_1['vl_pos'] = np.reshape(vl_pos, (-1, ))
            param_1['vl_clr'] = vl_clr
            save_render_params(dict(param_0.items()+param_1.items()),
                               output_dir, img_ind)

            # rotate by 90 degree along y-axis (vertical axis)
            mesh = util.rotate_model_in_place(mesh, 0, np.pi/2, 0)
            smpl = util.rotate_model_in_place(smpl, 0, np.pi/2, 0)

            pb.count()  # update progress bar

    pb.end()


if __name__ == '__main__':
    main()
