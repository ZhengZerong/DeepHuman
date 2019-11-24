from __future__ import division, print_function

import numpy as np
import cv2 as cv
import scipy.ndimage.interpolation as sii
import os
import signal

from DataUtil.VoxelizerUtil import load_volume_from_mat, rotate_volume
from Constants import consts

import threading
import sys
if sys.version_info >= (2, 0):
    print (sys.version)
    import Queue as queue
if sys.version_info >= (3, 0):
    print (sys.version)
    import queue


class DataLoader(threading.Thread):
    def __init__(self,
                 batch_size,
                 data_dir,
                 data_indices,
                 vol_res_x_w=128, vol_res_x_h=192,
                 vol_res_y_w=128, vol_res_y_h=192,
                 augmentation=True):
        super(DataLoader, self).__init__()
        if vol_res_x_h != 192 or vol_res_x_w != 128 or vol_res_y_h != 192 or vol_res_y_w != 128:
            print('Unsupported resolution!')
            raise ValueError

        self.batch_index = 0

        self.batch_size = batch_size
        self.vox_res_x = (vol_res_x_w, vol_res_x_h, vol_res_x_w)
        self.vox_res_y = (vol_res_y_w, vol_res_y_h, vol_res_y_w)

        self.data_dir = data_dir
        self.data_num = len(data_indices)
        self.data_indices = np.copy(data_indices)
        self.augmentation = augmentation
        self.reshuffle_indices()

        if augmentation:
            max_idx = np.max(self.data_indices)
            self.alpha = np.random.rand(max_idx+1)*0.3 + 0.85     # random from [0.9, 1.1]
            self.beta = np.random.rand(max_idx+1)*0.3 - 0.15    # random from [-0.05, 0.05]
            self.crop_size = np.random.randint(0, 20, (max_idx+1, 4))
            self.movement = np.random.randint(0, 11, (max_idx+1, 3)) - 5

        self.queue = queue.Queue(8)
        self.stop_queue = False

        self.total_batch_num = int(len(self.data_indices) // self.batch_size)

    def reshuffle_indices(self):
        self.batch_index = 0
        np.random.shuffle(self.data_indices)
        if self.augmentation:
            max_idx = np.max(self.data_indices)
            self.alpha = np.random.rand(max_idx+1)*0.3 + 0.85     # random from [0.9, 1.1]
            self.beta = np.random.rand(max_idx+1)*0.3 - 0.15    # random from [-0.05, 0.05]
            self.crop_size = np.random.randint(0, 20, (max_idx+1, 4))
            self.movement = np.random.randint(0, 11, (max_idx+1, 3)) - 5

    def load_volume(self, idx):
        volume_id = idx // 4 * 4
        view_id = idx - volume_id
        if consts.fill:
            volume = load_volume_from_mat('%s/voxel2/voxel_%08d.mat' % (self.data_dir, volume_id))
        else:
            volume = load_volume_from_mat('%s/voxel/voxel_%08d.mat' % (self.data_dir, volume_id))

        mesh_volume = rotate_volume(volume['mesh_volume'], view_id)
        smpl_v_volume = rotate_volume(volume['smpl_v_volume'], view_id)

        # convert from WHD format to DHW format (as required by tensorflow)
        mesh_volume = np.transpose(mesh_volume, (2, 1, 0))
        smpl_v_volume = np.transpose(smpl_v_volume, (2, 1, 0, 3))

        # flip upside down
        mesh_volume = np.flip(mesh_volume, axis=1)
        smpl_v_volume = np.flip(smpl_v_volume, axis=1)

        # if self.augmentation:
        #     movement = self.movement[idx, :]
        #     x_m, y_m, z_m = movement[0], movement[1], movement[2]
        #     smpl_v_volume = sii.shift(smpl_v_volume, (0, x_m, y_m, 0), cval=0)

        return smpl_v_volume, mesh_volume

    @staticmethod
    def resize_and_crop_img(img):
        img = cv.resize(img, (2*consts.dim_h, 2*consts.dim_h))
        edg = (2*consts.dim_h - 2*consts.dim_w) // 2
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        img = img[:, edg:-edg, :]
        return img

    def load_normal_maps(self, idx):
        volume_id = idx // 4 * 4
        view_id = idx - volume_id
        normal_0_id = volume_id + view_id
        normal_1_id = volume_id + (view_id+1) % 4
        normal_2_id = volume_id + (view_id+2) % 4
        normal_3_id = volume_id + (view_id+3) % 4

        normal_0_frame = cv.imread('%s/normal/normal_%08d.png' % (self.data_dir, normal_0_id), cv.IMREAD_UNCHANGED)
        normal_1_frame = cv.imread('%s/normal/normal_%08d.png' % (self.data_dir, normal_1_id), cv.IMREAD_UNCHANGED)
        normal_2_frame = cv.imread('%s/normal/normal_%08d.png' % (self.data_dir, normal_2_id), cv.IMREAD_UNCHANGED)
        normal_3_frame = cv.imread('%s/normal/normal_%08d.png' % (self.data_dir, normal_3_id), cv.IMREAD_UNCHANGED)

        normal_0_frame = np.float32(self.resize_and_crop_img(normal_0_frame))/32767.5-1.0
        normal_1_frame = np.float32(self.resize_and_crop_img(normal_1_frame))/32767.5-1.0
        normal_2_frame = np.float32(self.resize_and_crop_img(normal_2_frame))/32767.5-1.0
        normal_3_frame = np.float32(self.resize_and_crop_img(normal_3_frame))/32767.5-1.0

        #TODO: for testing
        # following code pass test
        # res = self.sess.run(gt_n, feed_dict={self.X: test_smpl_v_volumes[0],
        #                                      self.Y: test_mesh_volumes[0],
        #                                      self.R: test_conc_imgs[0][:, :, :, :6]})
        # first_n = res[0, :, :, :]
        # first_l = first_n[:, :, 0]*first_n[:, :, 0] + first_n[:, :, 1]*first_n[:, :, 1] + first_n[:, :, 2]*first_n[:, :, 2]
        # first_n = first_n / np.sqrt(np.expand_dims(first_l, axis=-1))
        # first_n_ = test_conc_imgs[0][0, :, :, 10:13]
        # first_n_[:, :, 0] *= -1
        # first_n_[:, :, 2] *= -1
        # first_n_ = cv.resize(first_n_, (first_n.shape[1], first_n.shape[0]))
        # cv.imwrite('./first_n.png', np.uint16(first_n*32767.5+32767.5))
        # cv.imwrite('./first_n_.png', np.uint16(first_n_*32767.5+32767.5))
        #
        # pdb.set_trace()

        for n in [normal_0_frame, normal_1_frame, normal_2_frame, normal_3_frame]:
            # n[:, :, 0] *= -1.0
            n[:, :, 2] *= -1.0

        for n in [normal_0_frame, normal_1_frame]:
            n[:, :, 0] *= -1.0

        normal_2_frame = np.fliplr(normal_2_frame)
        normal_3_frame = np.fliplr(normal_3_frame)

        return normal_0_frame, normal_1_frame, normal_2_frame, normal_3_frame

    def load_mask(self, idx):
        volume_id = idx // 4 * 4
        view_id = idx - volume_id
        mask_0_id = volume_id + view_id
        mask_1_id = volume_id + (view_id+1) % 4
        mask_2_id = volume_id + (view_id+2) % 4
        mask_3_id = volume_id + (view_id+3) % 4

        mask_0_frame = cv.imread('%s/mask/mask_%08d.png' % (self.data_dir, mask_0_id), cv.IMREAD_UNCHANGED)
        mask_1_frame = cv.imread('%s/mask/mask_%08d.png' % (self.data_dir, mask_1_id), cv.IMREAD_UNCHANGED)
        mask_2_frame = cv.imread('%s/mask/mask_%08d.png' % (self.data_dir, mask_2_id), cv.IMREAD_UNCHANGED)
        mask_3_frame = cv.imread('%s/mask/mask_%08d.png' % (self.data_dir, mask_3_id), cv.IMREAD_UNCHANGED)

        mask_0_frame = mask_0_frame/255
        mask_1_frame = mask_1_frame/255
        mask_2_frame = mask_2_frame/255
        mask_3_frame = mask_3_frame/255

        # kernel = np.ones((5, 5), np.uint8)
        # mask_0_frame = cv.erode(mask_0_frame/255, kernel, iterations=1)
        # mask_1_frame = cv.erode(mask_1_frame/255, kernel, iterations=1)
        # mask_2_frame = cv.erode(mask_2_frame/255, kernel, iterations=1)
        # mask_3_frame = cv.erode(mask_3_frame/255, kernel, iterations=1)

        mask_0_frame = np.float32(self.resize_and_crop_img(mask_0_frame))
        mask_1_frame = np.float32(self.resize_and_crop_img(mask_1_frame))
        mask_2_frame = np.float32(self.resize_and_crop_img(mask_2_frame))
        mask_3_frame = np.float32(self.resize_and_crop_img(mask_3_frame))

        mask_2_frame = np.fliplr(mask_2_frame)
        mask_3_frame = np.fliplr(mask_3_frame)

        return mask_0_frame, mask_1_frame, mask_2_frame, mask_3_frame

    def load_color_img(self, idx):
        img = cv.cvtColor(cv.imread('%s/color/color_%08d.jpg' % (self.data_dir, idx)), cv.COLOR_BGR2RGB)
        img = np.float32(img)/255.0

        if self.augmentation:
            alpha = self.alpha[idx]     # random from [0.9, 1.1]
            beta = self.beta[idx]    # random from [-0.05, 0.05]

            img = alpha * img + beta  # random brightness and contrast
            img = np.clip(img, 0.0, 1.0)

        img = self.resize_and_crop_img(img)
        # if self.augmentation:
        #     crop_size = self.crop_size[idx, :]
        #     img[0:crop_size[0], :, :] = 0.
        #     img[crop_size[1]:, :, :] = 0.
        #     img[:, 0:crop_size[2], :] = 0.
        #     img[:, crop_size[3]:, :] = 0.

        return img

    def load_vmap(self, idx):
        img = cv.cvtColor(cv.imread('%s/vmap/vmap_%08d.png' % (self.data_dir, idx)), cv.COLOR_BGR2RGB)
        img = np.float32(img)/255.0
        img = self.resize_and_crop_img(img)
        # if self.augmentation:
        #     movement = self.movement[idx, :]
        #     x_m, y_m, z_m = movement[0], movement[1], movement[2]
        #     img = sii.shift(img, (x_m, y_m, 0), cval=0)

        return img

    def load_tuple(self, idx):
        smpl_v_volume, mesh_volume = self.load_volume(idx)
        mesh_volume = np.expand_dims(mesh_volume, axis=-1)  # expand to [x_dim, y_dim, z_dim, channel] format

        n0, n1, n2, n3 = self.load_normal_maps(idx)
        m0, m1, m2, m3 = self.load_mask(idx)

        v = self.load_vmap(idx)
        c = self.load_color_img(idx)

        conc_img = np.concatenate((c, v, m0, m1, m2, m3, n0, n1, n2, n3), axis=-1)
        return conc_img, smpl_v_volume, mesh_volume

    def load_tuple_batch(self, indices):
        assert len(indices) == self.batch_size
        conc_imgs, smpl_v_volumes, mesh_volumes = [], [], []
        for idx in indices:

            conc_img, smpl_v_volume, mesh_volume = self.load_tuple(idx)
            while np.sum(mesh_volume) < 2e4:
                print('skipped one data. ')
                idx += 1
                conc_img, smpl_v_volume, mesh_volume = self.load_tuple(idx)

            conc_imgs.append(conc_img)
            smpl_v_volumes.append(smpl_v_volume)
            mesh_volumes.append(mesh_volume)

        conc_imgs = np.asarray(conc_imgs)
        smpl_v_volumes = np.asarray(smpl_v_volumes)
        mesh_volumes = np.asarray(mesh_volumes)
        return conc_imgs, smpl_v_volumes, mesh_volumes

    def load_tuple_next_batch(self):
        start_idx = self.batch_size * self.batch_index
        end_idx = self.batch_size * (self.batch_index + 1)
        data_indices = self.data_indices[start_idx:end_idx]
        self.batch_index += 1

        conc_imgs, smpl_v_volumes, mesh_volumes = self.load_tuple_batch(data_indices)
        return data_indices, conc_imgs, smpl_v_volumes, mesh_volumes

    def run(self):
        while not self.stop_queue:
            ## train
            if not self.queue.full():
                if self.batch_index>=self.total_batch_num:
                    self.reshuffle_indices()
                    # print ('shuffle')
                data_indices, conc_imgs, smpl_v_volumes, mesh_volumes = self.load_tuple_next_batch()
                self.queue.put((data_indices, conc_imgs, smpl_v_volumes, mesh_volumes))
