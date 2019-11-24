from __future__ import division, print_function

import os
import opendr
import numpy as np
import chumpy as ch
import cv2 as cv
import scipy.io as sio

from DataUtil.ObjIO import load_obj_data_binary, save_obj_data_binary
from DataUtil.CommonUtil import rotate_model_in_place
from opendr.lighting import LambertianPointLight, SphericalHarmonics, VertNormals
from opendr.renderer import ColoredRenderer, TexturedRenderer, BoundaryRenderer, DepthRenderer
from opendr.camera import ProjectPoints, ProjectPoints3D

dim_h = 192
dim_w = 128
hb_ratio = dim_w / dim_h
voxel_size = 1.0 / dim_h

# camera parameters
w, h = 1024, 1024
flength = 5000
trans = np.array([0, 0, flength / w * 2.0])


def main(mesh_list, out_list, scale=1.0, move_scale=True):
    assert len(mesh_list) == len(out_list)
    for mesh_file, out_file in zip(mesh_list, out_list):
        mesh = load_obj_data_binary(mesh_file)
        if move_scale:  # move to center and scale to unit bounding box
            mesh['v'] = (mesh['v'] - np.array([128, -192, 128]) + 0.5) * voxel_size

        if not ('vn' in mesh and mesh['vn'] is not None):
            mesh['vn'] = np.array(VertNormals(f=mesh['f'], v=mesh['v']))

        V = ch.array(mesh['v']) * scale
        V -= trans

        C = np.ones_like(mesh['v'])
        C *= np.array([186, 212, 255], dtype=np.float32) / 255.0
        # C *= np.array([158, 180, 216], dtype=np.float32) / 250.0
        C = np.minimum(C, 1.0)
        A = np.zeros_like(mesh['v'])
        A += LambertianPointLight(f=mesh['f'], v=V, vn=-mesh['vn'], num_verts=len(V),
                                  light_pos=np.array([0, -50, -50]),
                                  light_color=np.array([1.0, 1.0, 1.0]),
                                  vc=C)

        cam_t, cam_r = ch.array((0, 0, 0)), ch.array((3.14, 0, 0))
        U = ProjectPoints(v=V, f=[flength, flength], c=[w / 2., h / 2.], k=ch.zeros(5), t=cam_t,
                          rt=cam_r)
        rn = ColoredRenderer(camera=U, v=V, f=mesh['f'], vc=A,
                             bgcolor=np.array([1.0, 0.0, 0.0]),
                             frustum={'width': w, 'height': h, 'near': 0.1, 'far': 20})

        img = np.asarray(rn)[:, :, (2, 1, 0)]
        msk = np.sum(np.abs(img - np.array([[[0, 0, 1.0]]], dtype=np.float32)), axis=-1,
                     keepdims=True)
        msk[msk > 0] = 1
        img = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        msk = cv.resize(msk, (msk.shape[1] // 2, msk.shape[0] // 2), interpolation=cv.INTER_AREA)
        msk[msk < 1] = 0
        msk = msk[:, :, np.newaxis]
        img = np.concatenate([img, msk], axis=-1)
        cv.imshow('render3', img)
        cv.waitKey(3)
        cv.imwrite(out_file, np.uint8(img * 255))


def post_process(mesh_list, out_list):
    assert len(mesh_list) == len(out_list)
    for mesh_file, out_file in zip(mesh_list, out_list):
        mesh = load_obj_data_binary(mesh_file)
        out = cv.imread(out_file)
        # msk = 255 - out[:, :, 3]
        # out = out[:, :, :3]
        # out[:, :, 2] += msk
        param = sio.loadmat(
            mesh_file.replace('img_crop__volume_out_out_detailed.obj', 'corp_param.jpg.mat'))
        top = int(param['top'])
        left = int(param['left'])
        rows_ = int(param['rows_'])
        cols_ = int(param['cols_'])
        min_r_ = int(param['min_r_'])
        min_c_ = int(param['min_c_'])
        max_r_ = int(param['max_r_'])
        max_c_ = int(param['max_c_'])
        out_resize = out[top:(top + rows_), left:(left + cols_), :]
        out_crop = cv.resize(out_resize, (max_c_ - min_c_, max_r_ - min_r_))
        out_orig = np.ones((424, 512, 3), dtype=np.uint8) * np.array([[[0, 0, 255]]],
                                                                     dtype=np.uint8)
        out_orig[min_r_:max_r_, min_c_:max_c_, :] = out_crop
        cv.imshow('render3', out_orig)
        cv.waitKey(3)
        cv.imwrite(out_file[:-4] + '_orig.png', np.uint8(out_orig))


def post_process_2(mesh_list, out_list):
    assert len(mesh_list) == len(out_list)
    for mesh_file, out_file in zip(mesh_list, out_list):
        mesh = load_obj_data_binary(mesh_file)
        mesh['v'] = (mesh['v'] - np.array([128, -192, 128]) + 0.5) * voxel_size
        mesh['v'][:, 1:] *= -1.0
        mesh['vn'][:, 1:] *= -1.0
        mesh['f'] = mesh['f'][:, (1, 0, 2)]
        mesh['vc'] = np.array([])
        # mesh['vn'] = np.array(VertNormals(f=mesh['f'], v=mesh['v']))
        save_obj_data_binary(mesh, mesh_file[:-4] + '_orig_cam.obj')

        # test weak persepctive projection
        out = cv.imread(out_file)
        for v in mesh['v']:
            u = int(np.round(v[0] * 256 + 256))
            v = int(np.round(v[1] * 256 + 256))
            u = max(0, min(511, u))
            v = max(0, min(423, v))
            out[v, u, :] = np.array([255, 255, 255], dtype=np.uint8)
        cv.imwrite(out_file[:-4] + '_test_proj.png', out)


def post_process_3(mesh_list, out_list):
    out_folder = './testing_results/test_for_shunsuke/results/'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    assert len(mesh_list) == len(out_list)
    for mesh_file, out_file in zip(mesh_list, out_list):
        mesh_file = mesh_file[:-4] + '_orig_cam.obj'
        render_file = out_file
        # fd, _ = os.path.split(mesh_file)
        # fd, sub_sub_fd = os.path.split(fd)
        # _, sub_fd = os.path.split(fd)
        #
        # if not os.path.exists(os.path.join(out_folder, sub_fd)):
        #     os.mkdir(os.path.join(out_folder, sub_fd))
        # if not os.path.exists(os.path.join(out_folder, sub_fd, sub_sub_fd)):
        #     os.mkdir(os.path.join(out_folder, sub_fd, sub_sub_fd))
        # os.system('cp %s %s' % (mesh_file, os.path.join(out_folder, sub_fd, sub_sub_fd, 'result_mesh.obj')))
        # os.system('cp %s %s' % (render_file, os.path.join(out_folder, sub_fd, sub_sub_fd, 'result_render.png')))

        _, fname = os.path.split(mesh_file)
        os.system(
            'cp %s %s' % (mesh_file, os.path.join(out_folder, '%s_result_mesh.obj' % fname[:15])))
        os.system('cp %s %s' % (
        render_file, os.path.join(out_folder, '%s_result_render.png' % fname[:15])))


if __name__ == '__main__':
    # root_dir = './testing_results/vid_natalia/'
    # prefix_list = ['frame_c_0_f_%d' % i for i in range(1579, 1593)]
    # prefix_list += ['frame_c_0_f_%d' % i for i in range(1600, 1614)]
    # prefix_list += ['frame_c_0_f_%d' % i for i in range(1619, 1645)]
    # mesh_list = [os.path.join(root_dir, p + '__volume_out_out_detailed.obj')
    #              for p in prefix_list]
    # out_list = [os.path.join(root_dir, p + '_render.png')
    #             for p in prefix_list]

    # root_dir = './testing_results/vid_pablo/'
    # prefix_list = ['cam2_%06d' % i for i in range(0, 156)]
    # mesh_list = [os.path.join(root_dir, p + '__volume_out_out_detailed.obj')
    #              for p in prefix_list]
    # out_list = [os.path.join(root_dir, p + '_render.png')
    #             for p in prefix_list]

    # root_dir = './testing_results/vid_nadia/'
    # prefix_list = ['frame_%04d' % i for i in range(564, 901)]
    # mesh_list = [os.path.join(root_dir, p + '__volume_out_out_detailed.obj')
    #              for p in prefix_list]
    # out_list = [os.path.join(root_dir, p + '_render.png')
    #             for p in prefix_list]
    import glob

    mesh_list = glob.glob(
        './testing_results/test_for_shunsuke/Archive/*detailed.obj')
    out_list = [p + '_rendered.png' for p in mesh_list]
    # main(mesh_list, out_list)
    # post_process_2(mesh_list, out_list)
    post_process_3(mesh_list, out_list)
