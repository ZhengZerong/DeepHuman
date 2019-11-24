from __future__ import absolute_import, division, print_function
import os
import string
import random
from subprocess import call
import numpy as np
import chumpy as ch
import cv2 as cv
import scipy.io as sio
from scipy import ndimage
import zipfile

import CommonUtil as util
import DataUtil.CommonUtil as dutil
import DataUtil.ObjIO as objio
import DataUtil.PlyIO as plyio
import DataUtil.VoxelizerUtil as voxel_util
import DataUtil.renderers as renderers

dim_h = 192
dim_w = 128
voxel_size = 1.0 / dim_h
img2smpl_dir = 'path/to/Img2Smpl'
VOXELIZER_PATH = './voxelizer/build/bin'
preprocess_size = 384
crop_orig_img = False


def preprocess_image(img_dir):
    log_str = ''

    img = cv.imread(img_dir, cv.IMREAD_UNCHANGED)
    h, w = img.shape[0], img.shape[1]
    log_str += 'image size: %d, %d\n' % (h, w)

    h_c, w_c = h//2, w//2
    s = max(h, w) // 2
    img_pad = np.zeros((2*s, 2*s, 3))
    img_pad[(s-h_c):(s+h_c), (s-w_c):(s+w_c), :] = img[:h_c*2, :w_c*2, :]
    img_crop = img_pad
    img_crop = cv.resize(img_crop, (dim_h*2, dim_h*2))
    log_str += 'image padding/croping: s=%d, h_c=%d, w_c=%d\n' % (s, h_c, w_c)

    img_crop = cv.resize(img_crop, (preprocess_size, preprocess_size))
    cv.imwrite(img_dir[:-4] + '_color.png', img_crop)
    log_str += 'image saved to ' + img_dir[:-4] + '_color.png' + '\n'
    print(log_str)
    return log_str


def run_im2smpl(img_dir):
    log_str = ''
    tmp_folder = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    tmp_path = os.path.join(img2smpl_dir, tmp_folder)
    curr_path = os.getcwd()

    util.safe_mkdir(tmp_path)

    sh_file_str = ''
    sh_file_str += '#!/usr/local/bin/bash\n'
    sh_file_str += 'cp %s %s\n' % (img_dir[:-4] + '_color.png',
                                   os.path.join(tmp_path, 'test_img.png'))
    sh_file_str += 'cd ' + img2smpl_dir + '\n'
    if crop_orig_img:
        sh_file_str += 'python2 main.py --img_file %s --out_dir %s \n' \
                       % (os.path.join(tmp_path, 'test_img.png'), tmp_path)
    else:
        sh_file_str += 'python2 main_wo_cropping.py --img_file %s --out_dir %s \n' \
                       % (os.path.join(tmp_path, 'test_img.png'), tmp_path)

    sh_file_str += 'cd ' + curr_path + '\n'
    sh_file_str += 'mv %s %s\n' % (os.path.join(tmp_path, 'test_img.png.final.txt'),
                                   img_dir[:-4] + '_final.txt')
    sh_file_str += 'mv %s %s\n' % (os.path.join(tmp_path, 'test_img.png.smpl.obj'),
                                   img_dir[:-4] + '_smpl.obj')
    sh_file_str += 'mv %s %s\n' % (os.path.join(tmp_path, 'test_img.png.smpl_proj.png'),
                                   img_dir[:-4] + '_smpl_proj.png')
    sh_file_str += 'cp %s %s\n' % (os.path.join(tmp_path, 'test_img.png'),
                                   img_dir[:-4] + '_color.png')     # copies the cropped image back
    sh_file_str += 'rm -rf ' + tmp_path + '\n'

    sh_fname = ''.join(random.sample(string.ascii_letters + string.digits, 8)) + '.sh'

    with open(os.path.join('./', sh_fname), 'w') as fp:
        fp.write(sh_file_str)
    log_str += 'will run the following commands ------------\n'
    log_str += sh_file_str
    log_str += 'end of commend -----------------------------\n'
    print(log_str)

    call(['sh', os.path.join('./', sh_fname)])
    os.remove(os.path.join('./', sh_fname))
    return log_str


def postprocess_im2smpl_result(img_dir):
    log_str = ''

    # load and postprocess SMPL model (change camera model)
    with open(img_dir[:-4] + '_final.txt', 'r') as fp:
        lines = fp.readlines()
        line0_data = lines[0].split(' ')
        cam_tx = float(line0_data[0])
        cam_ty = float(line0_data[1])
        cam_tz = float(line0_data[2])

    img = cv.imread(img_dir[:-4] + '_color.png')
    cam_f = 5000.0
    cam_s = cam_f / (0.5 * img.shape[0] * cam_tz)

    smpl = objio.load_obj_data(img_dir[:-4] + '_smpl.obj')
    smpl['v'][:, 0] = (smpl['v'][:, 0] + cam_tx) * cam_s
    smpl['v'][:, 1] = (smpl['v'][:, 1] + cam_ty) * cam_s
    smpl['v'][:, 2] = smpl['v'][:, 2] * cam_s
    smpl['v'] *= 0.5

    dutil.flip_axis_in_place(smpl, 1, -1, -1)
    objio.save_obj_data_binary(smpl, img_dir[:-4] + '_smpl_2.obj')

    log_str += 'cam_tx = %f, cam_ty = %f, cam_s = %f\n' \
               % (cam_tx, cam_ty, cam_s)
    log_str += 'save to ' + img_dir[:-4] + '_smpl_2.obj\n'
    return log_str


def prepare_network_input_color(img_dir):
    log_str = ''
    img_size = 384

    # prepare input color for the network
    img = cv.imread(img_dir[:-4] + '_color.png')

    # gamma = 0.8
    # gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    # gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # img = cv.LUT(img, gamma_table)

    img = cv.resize(img, (img_size, img_size))
    cv.imwrite(img_dir[:-4] + '_color.png', img)
    log_str += 'save to ' + img_dir[:-4] + '_color.png\n'
    return log_str


def prepare_network_input_semantic(img_dir, zip_intermedia_results=True):
    log_str = ''
    img_size = 384

    # create semantic map
    smpl = objio.load_obj_data_binary(img_dir[:-4] + '_smpl_2.obj')
    smpl_volume = voxel_util.voxelize_2(img_dir[:-4] + '_smpl_2.obj',
                                        dim_h, dim_w, VOXELIZER_PATH)
    smpl_volume = voxel_util.binary_fill_from_corner_3D(smpl_volume)
    smpl_v_volume = voxel_util.calc_vmap_volume(smpl_volume, smpl['v'],
                                                dim_h, dim_w, voxel_size)
    mesh_volume = np.zeros((dim_w, dim_h, dim_w), dtype=np.float32)
    sio.savemat(img_dir[:-4] + '_volume.mat',
                {'mesh_volume': mesh_volume, 'smpl_v_volume': smpl_v_volume},
                do_compression=True)

    # render semantic map
    smpl_v_ = renderers.compress_along_z_axis_single(smpl['v'])
    u = renderers.project_vertices(smpl_v_, img_size, img_size,
                                   cam_r=ch.array((3.14, 0, 0)),
                                   cam_t=ch.array((0, 0, 0)))
    smpl_vc = dutil.get_smpl_semantic_code()
    v_map = renderers.render_color_model_without_lighting(img_size, img_size,
                                                          smpl_v_, smpl_vc,
                                                          smpl['f'], u,
                                                          bg_img=None)
    v_map = np.float32(np.copy(v_map))
    cv.imwrite(img_dir[:-4] + '_vmap.png', np.uint8(v_map*255))

    # test orthogonal projection from SMPL's volume
    smpl_volume_proj = np.max(smpl_volume, axis=-1)
    smpl_volume_proj = np.flipud(np.transpose(smpl_volume_proj))
    smpl_volume_proj = cv.resize(np.uint8(smpl_volume_proj) * 255, (256, 384))
    cv.imwrite(img_dir[:-4] + '_test_proj.png', smpl_volume_proj)

    # save smpl semantic volume for visualization
    voxel_util.save_v_volume(smpl_v_volume, img_dir[:-4] + '_v_volume.obj',
                             dim_h, dim_w, voxel_size)

    suffixes = ['_final.txt', '_smpl.obj', '_smpl_2.obj', '_smpl_proj.png',
                '_test_proj.png', '_v_volume.obj']
    if zip_intermedia_results:
        z = zipfile.ZipFile(img_dir[:-4] + '_intermediate_prepare.zip', 'w')
        for suffix in suffixes:
            z.write(img_dir[:-4] + suffix)
            os.remove(img_dir[:-4] + suffix)
        z.close()

    print(log_str)
    return log_str


def main(img_dir):
    preprocess_image(img_dir)
    run_im2smpl(img_dir)
    postprocess_im2smpl_result(img_dir)
    prepare_network_input_color(img_dir)
    prepare_network_input_semantic(img_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='path to image file')
    args = parser.parse_args()

    img_dir = args.file
    if not (img_dir.endswith('.png') or img_dir.endswith('.jpg')):
        print('Unsupport image format!!!')
        raise ValueError

    main(img_dir)
