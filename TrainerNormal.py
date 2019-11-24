from __future__ import division, print_function, absolute_import

import os
import time
import shutil
import numpy as np
import cv2 as cv
import glob
import scipy.io as sio

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers

from Ops import Ops
from DataLoaderNormal import DataLoader
from CommonUtil import logger, safe_rm_mkdir, safe_mkdir
from Constants import consts

log = logger.write


class Trainer(object):
    def __init__(self, sess):
        self.sess = sess

    def train(self,
              dataset_dir,                  # path to dataset
              dataset_training_indices,     # data starting index
              dataset_testing_indices,      # data ending index
              results_dir='./results/results_depth_multi_normal3',      # directory to stored the results
              graph_dir='./results/graph_depth_multi_normal3',          # directory as tensorboard working space
              batch_size=4,                 # batch size
              epoch_num=9,                 # epoch number
              first_channel=8,
              bottle_width=4,
              dis_reps=1,
              mode='retrain',               # training mode: 'retrain' or 'finetune'
              pre_model_dir=None):          # directory to pre-trained model
        """
        Train
        construct the network, data loader and loss function accroding to the argument
        """
        assert batch_size > 1  # tf.squeeze is used, so need to make sure that the batch dim won't be removed

        self._setup_result_folder(mode, results_dir, graph_dir)
        # logger.set_log_file(results_dir + '/log.txt')

        # setups data loader
        data_loader_num = 4
        data_loader_num_half = data_loader_num //2
        data_loaders, val_data_loader = self._setup_data_loader(data_loader_num, batch_size, dataset_dir, dataset_training_indices, dataset_testing_indices)
        batch_num = len(dataset_training_indices)//batch_size
        test_batch_num = 16 // batch_size
        log('#epoch = %d, #batch = %d' % (epoch_num, batch_num))

        # loads some testing data for visualization and supervision
        test_conc_imgs, test_smpl_v_volumes, test_mesh_volumes = [], [], []
        safe_rm_mkdir(results_dir + '/test_gt')
        for i in range(test_batch_num):
            _, conc_imgs, smpl_v_volumes, mesh_volumes = val_data_loader.queue.get()
            test_conc_imgs.append(conc_imgs)
            test_smpl_v_volumes.append(smpl_v_volumes)
            test_mesh_volumes.append(mesh_volumes)
            self._save_tuple(conc_imgs, smpl_v_volumes, mesh_volumes, results_dir+'/test_gt', i)

        # setups network and training loss
        self._build_network(batch_size, first_channel, bottle_width)
        loss_collection = self._build_loss(self.v_d[-1], self.Y, self.M_fv, self.M_sv,
                                           self.Ns, self.n_final, self.dis_real_out, self.dis_fake_out,
                                           lamb_sil=self.lamb_sil, lamb_nml_rf=self.lamb_nml,
                                           lamb_dis=self.lamb_dis, w=0.7)
        loss_keys = ['vol_loss', 'sil_loss', 'normal_loss', 'nr_loss', 'recon_loss', 'total_loss']

        # setups optimizer and visualizer
        recon_loss = loss_collection['recon_loss']
        nr_loss = loss_collection['nr_loss']
        total_loss = loss_collection['total_loss']
        dis_d_loss = loss_collection['dis_d_loss']
        recon_opt, nr_opt, all_opt, dis_opt = self._build_optimizer(self.lr, recon_loss, nr_loss, total_loss, dis_d_loss)
        merged_scalar_loss, writer = self._setup_summary(self.sess, graph_dir, loss_collection)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        saver = self._setup_saver(pre_model_dir)

        for epoch_id in range(epoch_num):
            log('Running epoch No.%d' % epoch_id)
            loss_log_str = ''
            for batch_id in range(batch_num):
                iter_id = epoch_id*batch_num+batch_id
                lrate = 1e-4 if epoch_id <= epoch_num/3*2 else 1e-5
                lrate_d = lrate * 0.1
                l_sil = consts.lamb_sil
                l_dis = consts.lamb_dis
                l_nml_rf = consts.lamb_nml_rf

                # training ==============================================================
                ind1, conc_imgs1, smpl_v_volumes1, mesh_volumes1 = data_loaders[iter_id % data_loader_num_half].queue.get()
                ind2, conc_imgs2, smpl_v_volumes2, mesh_volumes2 = data_loaders[iter_id % data_loader_num_half + data_loader_num_half].queue.get()
                conc_imgs = np.concatenate([conc_imgs1, conc_imgs2], axis=0)
                smpl_v_volumes = np.concatenate([smpl_v_volumes1, smpl_v_volumes2], axis=0)
                mesh_volumes = np.concatenate([mesh_volumes1, mesh_volumes2], axis=0)

                f_dict = self._construct_feed_dict(conc_imgs, smpl_v_volumes, mesh_volumes, l_sil, l_dis, l_nml_rf, lrate)
                f_dict_d = self._construct_feed_dict(conc_imgs, smpl_v_volumes, mesh_volumes, l_sil, l_dis, l_nml_rf, lrate_d)

                # if epoch_id <= epoch_num / 3:
                #     out = self.sess.run([recon_opt] + [loss_collection[lk] for lk in loss_keys] + [merged_scalar_loss], feed_dict=f_dict)
                #     loss_curr_list = out[1:-1]
                #     graph_results = out[-1]
                #
                # elif epoch_id <= epoch_num / 3 * 2:
                #     for _ in range(dis_reps):
                #         self.sess.run([dis_opt], feed_dict=f_dict_d)
                #     out = self.sess.run([nr_opt] + [loss_collection[lk] for lk in loss_keys] + [merged_scalar_loss],feed_dict=f_dict)
                #     loss_curr_list = out[1:-1]
                #     graph_results = out[-1]
                # else:
                if True:
                    for _ in range(dis_reps):
                        self.sess.run([dis_opt], feed_dict=f_dict_d)
                    out = self.sess.run([recon_opt, nr_opt] + [loss_collection[lk] for lk in loss_keys] + [merged_scalar_loss], feed_dict=f_dict)
                    loss_curr_list = out[2:-1]
                    graph_results = out[-1]

                writer.add_summary(graph_results, epoch_id * batch_num + batch_id)

                scale = 1
                log('Epoch %d, Batch %d: '
                    'vol_loss:%.4f, sil_loss:%.4f, normal_loss:%.4f, nr_loss:%.4f, '
                    'recon_loss:%.4f, total_loss:%.4f' %
                    (epoch_id, batch_id, loss_curr_list[0] * scale, loss_curr_list[1] * scale,
                     loss_curr_list[2] * scale, loss_curr_list[3] * scale,
                     loss_curr_list[4] * scale, loss_curr_list[5] * scale))

                # validation ===========================================================
                if iter_id % 5 == 0:
                    _, conc_imgs, smpl_v_volumes, mesh_volumes = val_data_loader.queue.get()
                    f_dict = self._construct_feed_dict(conc_imgs, smpl_v_volumes, mesh_volumes, l_sil, l_dis, l_nml_rf, lrate)
                    loss_val_curr = self.sess.run([loss_collection[lk] for lk in loss_keys], feed_dict=f_dict)

                    loss_log_str += ('%f %f %f %f %f %f ' % (loss_curr_list[0], loss_curr_list[1], loss_curr_list[2], 
                                                             loss_curr_list[3], loss_curr_list[4], loss_curr_list[5]))
                    loss_log_str += ('%f %f %f %f %f %f \n' % (loss_val_curr[0], loss_val_curr[1], loss_val_curr[2], 
                                                               loss_val_curr[3], loss_val_curr[4], loss_val_curr[5]))

            log('End of epoch. ')
            with open(os.path.join(results_dir, 'loss_log.txt'), 'a') as fp:
                fp.write(loss_log_str)

            if epoch_id > 0.5 * epoch_num:
                test_dir = os.path.join(results_dir, '%04d' % epoch_id)
                safe_rm_mkdir(test_dir)
                saver.save(self.sess, os.path.join(results_dir, 'model.ckpt'))

                # test the network and save the results
                for tbi in range(test_batch_num):
                    f_dict = {self.X: test_smpl_v_volumes[tbi], self.Y: test_mesh_volumes[tbi],
                              self.R: test_conc_imgs[tbi][:, :, :, :6]}
                    n0_p, n1_p, n2_p, n3_p = self.sess.run([self.n0_project, self.n1_project,
                                                            self.n2_project, self.n3_project],
                                                           feed_dict=f_dict)
                    nps = np.concatenate((n0_p, n1_p, n2_p, n3_p), axis=-1)

                    res = self.sess.run(self.v_out, feed_dict=f_dict)
                    res_n = self.sess.run(self.n_final, feed_dict=f_dict)
                    self._save_results_raw_training(res, res_n, nps, test_dir, tbi)

                # backup model
            if True:    # epoch_id % 10 == 0:
                saver.save(self.sess, os.path.join(results_dir, 'model.ckpt'))

        for data in data_loaders:
            data.stop_queue = True
        val_data_loader.stop_queue = True

    @staticmethod
    def _setup_result_folder(mode,
                             results_dir='./results/results_depth_multi_normal3',
                             graph_dir='./results/graph_depth_multi_normal3'):
        # create folders
        if mode == 'retrain':
            if os.path.exists(results_dir):
                log('Warning: %s already exists. It will be removed. ' % results_dir)
                shutil.rmtree(results_dir)
            if os.path.exists(graph_dir):
                log('Warning: %s already exists. It will be removed. ' % graph_dir)
                shutil.rmtree(graph_dir)
            safe_rm_mkdir(results_dir)
            safe_rm_mkdir(graph_dir)

        safe_rm_mkdir(results_dir + '/code_bk')
        pylist = glob.glob(os.path.join('./', '*.py'))
        for pyfile in pylist:
            shutil.copy(pyfile, results_dir + '/code_bk')

    @staticmethod
    def _setup_data_loader(data_loader_num, batch_size, dataset_dir,
                           dataset_training_indices, dataset_testing_indices):
        log('Constructing data loader...')
        log('#training_data =', len(dataset_training_indices))
        log('#testing_data =', len(dataset_testing_indices))

        data_loaders = []
        for _ in range(data_loader_num//2):
            data = DataLoader(batch_size//2, dataset_dir, dataset_training_indices, augmentation=True)
            data.daemon = True
            data.start()
            data_loaders.append(data)

        for _ in range(data_loader_num//2):
            data = DataLoader(batch_size//2, dataset_dir+'_twindom', np.asarray(range(16000)), augmentation=True)
            data.daemon = True
            data.start()
            data_loaders.append(data)

        val_data_loader = DataLoader(batch_size, dataset_dir, dataset_testing_indices, augmentation=False)
        val_data_loader.daemon = True
        val_data_loader.start()

        log('DataLoaders start. ')
        return data_loaders, val_data_loader

    def _construct_feed_dict(self, conc_imgs, smpl_v_volumes, mesh_volumes,
                             l_sil, l_dis, l_nml_rf, lrate):
        in_imgs = conc_imgs[:, :, :, :6]  # use only first 6 channels as input
        m0, m1 = conc_imgs[:, :, :, 6:7], conc_imgs[:, :, :, 7:8]
        n0, n1 = conc_imgs[:, :, :, 10:13], conc_imgs[:, :, :, 13:16]
        n2, n3 = conc_imgs[:, :, :, 16:19], conc_imgs[:, :, :, 19:22]
        f_dict = {self.X: smpl_v_volumes,
                  self.Y: mesh_volumes,
                  self.R: in_imgs,
                  self.M_fv: m0, self.M_sv: m1,
                  self.N0: n0, self.N1: n1, self.N2: n2, self.N3: n3,
                  self.lamb_sil: l_sil, self.lamb_dis: l_dis, self.lamb_nml: l_nml_rf,
                  self.lr: lrate}
        return f_dict

    def test(self,
             dataset_dir,           # path to testing dataset
             dataset_prefix_list,   # file name prefix of testing data
             pre_model_dir,         # path to pretrained model
             first_channel=8,
             bottle_width=4):  # directory to pre-trained model
        """
            Test
            construct the network, data loader and loss function accroding to the argument
        """
        log = logger.write
        batch_size = 1  # batch size

        self._build_network(batch_size, first_channel, bottle_width)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        log('Constructing saver...')
        saver = self._setup_saver(pre_model_dir)

        for dataset_prefix in dataset_prefix_list:
            prefix = dataset_dir + '/' + dataset_prefix
            img = cv.cvtColor(cv.imread(prefix + 'color.png'), cv.COLOR_BGR2RGB)
            # prefix = './TestingData/test_'
            # img = cv.cvtColor(cv.imread(prefix + 'input.jpg'), cv.COLOR_BGR2RGB)
            img = np.float32(img) / 255.0
            img = DataLoader.resize_and_crop_img(img)

            vmap = cv.cvtColor(cv.imread(prefix + 'vmap.png'), cv.COLOR_BGR2RGB)
            vmap = np.float32(vmap) / 255.0
            vmap = DataLoader.resize_and_crop_img(vmap)

            smpl_v_volume = sio.loadmat(prefix + 'volume.mat')
            smpl_v_volume = smpl_v_volume['smpl_v_volume']
            smpl_v_volume = np.transpose(smpl_v_volume, (2, 1, 0, 3))
            smpl_v_volume = np.flip(smpl_v_volume, axis=1)

            concat_in = np.concatenate((img, vmap), axis=-1)
            concat_in = np.expand_dims(concat_in, axis=0)
            smpl_v_volume = np.expand_dims(smpl_v_volume, axis=0)

            n0_p, n1_p, n2_p, n3_p = self.sess.run([self.n0_project, self.n1_project, self.n2_project, self.n3_project],
                                                   feed_dict={self.X: smpl_v_volume, self.R: concat_in})
            nps = np.concatenate((n0_p, n1_p, n2_p, n3_p), axis=-1)

            res = self.sess.run(self.v_out, feed_dict={self.X: smpl_v_volume, self.R: concat_in})
            res_n = self.sess.run(self.n_final, feed_dict={self.X: smpl_v_volume, self.R: concat_in})
            log('Testing results saved to', dataset_dir)
            self._save_results_raw_testing(res, res_n, nps, dataset_dir, dataset_prefix)

    def test_with_gt(self,
                     dataset_dir,  # path to dataset
                     dataset_testing_indices,  # data ending index
                     pre_model_dir,
                     output_dir,
                     first_channel=8,
                     bottle_width=4):  # directory to pre-trained model
        safe_mkdir(output_dir)
        loader = DataLoader(1, dataset_dir, dataset_testing_indices, augmentation=False)

        self._build_network(1, first_channel, bottle_width)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        saver = self._setup_saver(pre_model_dir)

        for i in dataset_testing_indices:
            conc_imgs, smpl_v_volumes, mesh_volumes = loader.load_tuple_batch([i])
            f_dict = self._construct_feed_dict(conc_imgs, smpl_v_volumes, mesh_volumes, 0,
                                               0, 0, 0)
            n0_p, n1_p, n2_p, n3_p = self.sess.run([self.n0_project, self.n1_project,
                                                    self.n2_project, self.n3_project],
                                                   feed_dict=f_dict)
            nps = np.concatenate((n0_p, n1_p, n2_p, n3_p), axis=-1)
            res, res_n = self.sess.run([self.v_out, self.n_final],
                                       feed_dict=f_dict)
            log('Testing results saved to ', output_dir)
            self._save_results_raw_testing(res, res_n, nps, output_dir, '%08d_' % i)

    def _build_network(self, batch_size, first_channel, bottle_width):
        """
        Builds the image-guided volume-to-volume network
        Warning: the network input format: BDHWC for volume, and BHWC for image
        """
        log('Constructing network...')

        with tf.name_scope('params'):
            self.lamb_sil = tf.placeholder(dtype=tf.float32)
            self.lamb_dis = tf.placeholder(dtype=tf.float32)
            self.lamb_nml = tf.placeholder(dtype=tf.float32)
            self.lr = tf.placeholder(dtype=tf.float32)

        with tf.name_scope('input'):
            self.X = tf.placeholder(shape=[batch_size, consts.dim_w, consts.dim_h, consts.dim_w, 3], dtype=tf.float32)
            self.Y = tf.placeholder(shape=[batch_size, consts.dim_w, consts.dim_h, consts.dim_w, 1], dtype=tf.float32)
            self.R = tf.placeholder(shape=[batch_size, 2*consts.dim_h, 2*consts.dim_w, 6], dtype=tf.float32)
            self.M_fv = tf.placeholder(shape=[batch_size, 2*consts.dim_h, 2*consts.dim_w, 1], dtype=tf.float32)
            self.M_sv = tf.placeholder(shape=[batch_size, 2*consts.dim_h, 2*consts.dim_w, 1], dtype=tf.float32)
            self.N0 = tf.placeholder(shape=[batch_size, 2 * consts.dim_h, 2 * consts.dim_w, 3], dtype=tf.float32)
            self.N1 = tf.placeholder(shape=[batch_size, 2 * consts.dim_h, 2 * consts.dim_w, 3], dtype=tf.float32)
            self.N2 = tf.placeholder(shape=[batch_size, 2 * consts.dim_h, 2 * consts.dim_w, 3], dtype=tf.float32)
            self.N3 = tf.placeholder(shape=[batch_size, 2 * consts.dim_h, 2 * consts.dim_w, 3], dtype=tf.float32)
            self.Ns = tf.concat([self.N0, self.N1, self.N2, self.N3], axis=-1)

        with tf.name_scope('network'):
            self.i_e = self._build_image_encoder(self.R, first_channel, bottle_width, logger.write)
            self.sft_a, self.sft_b = self._build_affine_params(self.i_e, logger.write)
            self.v_e = self._build_volume_encoder(self.X, first_channel, bottle_width, self.sft_a, self.sft_b, logger.write)
            self.v_d = self._build_volume_decoder(self.v_e, 1, consts.dim_w, self.sft_a, self.sft_b, logger.write)
            self.v_out = self.v_d[-1]
            self.d0, self.d1, self.d2, self.d3 = self._build_depth_projector(self.v_out)
            self.n0_project = self._build_normal_calculator(self.d0)
            self.n1_project = self._build_normal_calculator(self.d1)
            self.n2_project = self._build_normal_calculator(self.d2)
            self.n3_project = self._build_normal_calculator(self.d3)
            self.nr0 = self._build_normal_refiner(self.n0_project, self.R, logger.write)
            self.n_final_0 = self.nr0[-1]
            self.nr1, self.nr2, self.nr3 = self._build_normal_refiner2(self.n1_project, self.n2_project, self.n3_project, logger.write)
            self.n_final_1, self.n_final_2, self.n_final_3 = self.nr1[-1], self.nr2[-1], self.nr3[-1]
            self.n_final = tf.concat([self.n_final_0, self.n_final_1, self.n_final_2, self.n_final_3], axis=-1)
            self.dis_real_out, self.dis_fake_out = self._build_normal_discriminator(self.n_final, self.Ns, self.M_fv, self.M_sv, self.R, logger.write)

        log('The whole graph has %d trainable parameters' % Ops.get_variable_num(logger))

    @staticmethod
    def _build_image_encoder(R, first_channel, bottle_neck_w, print_fn=None):
        """
            Build the volume encoder
        """
        R_ = tf.image.resize_bilinear(R, (consts.dim_h, consts.dim_w))

        r_shape = R_.get_shape().as_list()
        r_w = r_shape[2]
        if print_fn is None:
            print_fn = print

        # calculate network parameters
        w_e = [r_w // 2]
        c_e = [first_channel]
        while w_e[-1] > bottle_neck_w:
            w_e.append(w_e[-1]//2)
            c_e.append(c_e[-1]*2)
        print_fn('-- Image encoder layers\' width', w_e)
        print_fn('-- Image encoder layers\' channel', c_e)
        layers = [R_]
        for c in c_e:
            with tf.variable_scope('i_e_%d' % (len(layers))):
                nin_shape = layers[-1].get_shape().as_list()
                net = slim.conv2d(layers[-1], c, [7, 7], 2, padding='SAME',
                                  weights_initializer=initializers.xavier_initializer(),
                                  weights_regularizer=None,
                                  rate=1, normalizer_fn=slim.batch_norm,
                                  activation_fn=tf.nn.leaky_relu, scope='conv0')
                print_fn('-- Image encoder layer %d:'%len(layers), nin_shape, '-->', net.get_shape().as_list())
                layers.append(net)

        return layers

    @staticmethod
    def _build_affine_params(E_i, print_fn=None):
        if print_fn is None:
            print_fn = print

        sft_a, sft_b = [], []
        for li in range(1, len(E_i)):
            with tf.variable_scope('a_p_%d' % (len(sft_a)+1)):
                nin_shape = E_i[li].get_shape().as_list()
                net_a = slim.conv2d(E_i[li], nin_shape[-1], [1, 1], 1, padding='SAME',
                                    weights_initializer=initializers.xavier_initializer(),
                                    weights_regularizer=None,
                                    rate=1, normalizer_fn=slim.batch_norm,
                                    activation_fn=tf.nn.leaky_relu, scope='conv0_pa')
                sft_a.append(net_a)
                net_b = slim.conv2d(E_i[li], nin_shape[-1], [1, 1], 1, padding='SAME',
                                    weights_initializer=initializers.xavier_initializer(),
                                    weights_regularizer=None,
                                    rate=1, normalizer_fn=slim.batch_norm,
                                    activation_fn=tf.nn.leaky_relu, scope='conv0_pb')
                sft_b.append(net_b)
                print_fn('-- SFT parameters layer %d:' % len(sft_a), nin_shape, '-->', net_a.get_shape().as_list())
        return sft_a, sft_b

    @staticmethod
    def _build_volume_encoder(X, frist_channel, bottle_neck_w,  sft_params_a, sft_params_b, print_fn=None):
        """
        Build the volume encoder
        """
        x_shape = X.get_shape().as_list()   # (batch, x_dim, y_dim, z_dim, channel)
        x_w = x_shape[1]
        if print_fn is None:
            print_fn = print

        # calculate network parameters
        w_e = [x_w//2]
        c_e = [frist_channel]
        while w_e[-1] > bottle_neck_w:
            w_e.append(w_e[-1]//2)
            c_e.append(c_e[-1]*2)
        print_fn('-- Volume encoder layers\' width', w_e)
        print_fn('-- Volume encoder layers\' channel', c_e)

        layers = [X]
        for ci, c in enumerate(c_e):
            with tf.variable_scope('v_e_%d' % len(layers)):
                nin_shape = layers[-1].get_shape().as_list()
                net = slim.conv3d(layers[-1], c, [7, 7, 7], 2, padding='SAME',
                                  weights_initializer=initializers.xavier_initializer(),
                                  weights_regularizer=None,
                                  rate=1, normalizer_fn=slim.batch_norm,
                                  activation_fn=tf.nn.leaky_relu, scope='conv0')
                net = Ops.featrue_affine(net, sft_params_a[ci], sft_params_b[ci])
                print_fn('-- Volume encoder layer %d:' % len(layers), nin_shape, '-->', net.get_shape().as_list())
                layers.append(net)
        return layers

    @staticmethod
    def _build_volume_decoder(layers_e, last_channel, out_w, sft_params_a, sft_params_b, print_fn=None):
        """
        Build the volume decoder
        """
        Z = layers_e[-1]
        z_shape = Z.get_shape().as_list()
        z_w = z_shape[1]
        z_c = z_shape[-1]
        if print_fn is None:
            print_fn = print

        # calculate network parameters
        w_d = [z_w*2]
        c_d = [z_c//2]
        while w_d[-1] < out_w:
            w_d.append(w_d[-1]*2)
            c_d.append(c_d[-1]//2)
        print_fn('-- Volume decoder layers\' width', w_d)
        print_fn('-- Volume decoder layers\' channel', c_d)

        layers = [Z]
        for ci, c in enumerate(c_d):
            with tf.variable_scope('v_d_%d' % len(layers)):
                if ci == 0:
                    net = layers[-1]
                else:
                    net = tf.concat([layers[-1], layers_e[-ci-1]], axis=-1)   # U-net structure
                nin_shape = net.get_shape().as_list()
                net = slim.conv3d_transpose(net, c, [7, 7, 7], 2, padding='SAME',
                                            weights_initializer=initializers.xavier_initializer(),
                                            weights_regularizer=None,
                                            normalizer_fn=slim.batch_norm,
                                            activation_fn=tf.nn.leaky_relu, scope='conv0')
                print_fn('-- Volume decoder layer %d:' % len(layers), nin_shape, '-->', net.get_shape().as_list())
                layers.append(net)

        with tf.variable_scope('v_d_out'):
            nin_shape = layers[-1].get_shape().as_list()
            net = slim.conv3d(layers[-1], last_channel, [1, 1, 1], 1, padding='SAME',
                              weights_initializer=initializers.xavier_initializer(),
                              weights_regularizer=None,
                              rate=1, normalizer_fn=None,
                              activation_fn=tf.nn.sigmoid, scope='conv0')       # output to (0, 1)
            print_fn('-- Volume decoder layer %d:' % len(layers), nin_shape, '-->', net.get_shape().as_list())
            layers.append(net)

        return layers

    @staticmethod
    def _build_sil_projector(volume):
        with tf.name_scope('projector'):
            vshape = volume.get_shape().as_list()
            v1 = tf.reshape(volume, (vshape[0], vshape[1], vshape[2], vshape[3]))   # remove last dim
            fv = tf.reduce_max(v1, axis=1)      # project along z-axis
            fv = tf.squeeze(fv)                 # remove z-dim
            fv = tf.expand_dims(fv, axis=-1)    # add channel dim

            sv = tf.reduce_max(v1, axis=3)      # project along x-axis
            sv = tf.squeeze(sv)                 # remove x-dim
            sv = tf.transpose(sv, (0, 2, 1))    # convert to HW format
            sv = tf.expand_dims(sv, axis=-1)    # add channel dim

        return fv, sv

    @staticmethod
    def _build_depth_projector(volume):
        with tf.name_scope('depth_projector'):
            vshape = volume.get_shape().as_list()
            v1 = tf.reshape(volume, (vshape[0], vshape[1], vshape[2], vshape[3]))  # remove last dim
            v1 = tf.sigmoid(9999*(v1-0.5))

            d_array = np.asarray(range(consts.dim_w), dtype=np.float32)
            d_array = (d_array - (consts.dim_w / 2) + 0.5) * consts.voxel_size

            d_array = tf.constant(d_array, dtype=tf.float32)

            # front view (view 0) projection (along z-axis)
            M = -99
            d_array_v0 = tf.reshape(d_array, (1, -1, 1, 1))    # BDHW
            depth_volume_0 = M*(1-v1) + d_array_v0 * v1
            depth_project_0 = tf.reduce_max(depth_volume_0, axis=1)    # max along D (Z) --> BHW
            depth_project_0 = tf.reshape(depth_project_0, (vshape[0], vshape[2], vshape[3], 1))

            # side view (view 1) projection (along x_axis)
            M = 99
            d_array_v1 = tf.reshape(d_array, (1, 1, 1, -1))
            depth_volume_1 = M*(1-v1) + d_array_v1 * v1
            depth_project_1 = tf.reduce_min(depth_volume_1, axis=3)    # min along W (X) --> BDH
            depth_project_1 = -depth_project_1
            depth_project_1 = tf.reshape(tf.transpose(depth_project_1, (0, 2, 1)), (vshape[0], vshape[2], vshape[3], 1))

            # back view (view 2) projection (along z-axis)
            M = 99
            depth_volume_2 = M*(1-v1) + d_array_v0 * v1
            depth_project_2 = tf.reduce_min(depth_volume_2, axis=1)    # max along D (Z) --> BHW
            depth_project_2 = -depth_project_2
            depth_project_2 = tf.reshape(depth_project_2, (vshape[0], vshape[2], vshape[3], 1))

            # size view (view 3) projection (along x-axis)
            M = -99
            depth_volume_3 = M*(1-v1) + d_array_v1 * v1
            depth_project_3 = tf.reduce_max(depth_volume_3, axis=3)    # min along W (X) --> BDH
            depth_project_3 = tf.reshape(tf.transpose(depth_project_3, (0, 2, 1)), (vshape[0], vshape[2], vshape[3], 1))

        return depth_project_0, depth_project_1, depth_project_2, depth_project_3

    @staticmethod
    def _build_normal_calculator(depth):
        d_shape = depth.get_shape().as_list()
        batch_sz = d_shape[0]
        img_h = d_shape[1]
        img_w = d_shape[2]

        w_array = np.asarray(range(consts.dim_w), dtype=np.float32)
        w_array = (w_array - (consts.dim_w / 2) + 0.5) * consts.voxel_size
        w_array = np.reshape(w_array, (1, 1, -1, 1))        # BHWC
        w_array = np.tile(w_array, (batch_sz, img_h, 1, 1))
        w_map = tf.constant(w_array, dtype=tf.float32)

        h_array = np.asarray(range(consts.dim_h), dtype=np.float32)
        h_array = (h_array - (consts.dim_h / 2) + 0.5) * consts.voxel_size
        h_array = np.reshape(h_array, (1, -1, 1, 1))        # BHWC
        h_array = np.tile(h_array, (batch_sz, 1, img_w, 1))
        h_map = tf.constant(h_array, dtype=tf.float32)

        # vmap = tf.concat([w_map, h_map, depth], axis=-1)

        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
        sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

        w_map_dx = tf.nn.conv2d(w_map, sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
        h_map_dx = tf.nn.conv2d(h_map, sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
        depth_dx = tf.nn.conv2d(depth, sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
        dx = tf.concat([w_map_dx, h_map_dx, depth_dx], axis=-1)

        w_map_dy = tf.nn.conv2d(w_map, sobel_y_filter, strides=[1, 1, 1, 1], padding='SAME')
        h_map_dy = tf.nn.conv2d(h_map, sobel_y_filter, strides=[1, 1, 1, 1], padding='SAME')
        depth_dy = tf.nn.conv2d(depth, sobel_y_filter, strides=[1, 1, 1, 1], padding='SAME')
        dy = tf.concat([w_map_dy, h_map_dy, depth_dy], axis=-1)

        normal = tf.cross(dy, dx)
        normal = normal / tf.norm(normal, axis=-1, keepdims=True)
        return normal

    @staticmethod
    def _build_normal_refiner(normal_0, rgb, print_fn=None):
        conc_d = tf.image.resize_bilinear(normal_0, (consts.dim_h * 2, consts.dim_w * 2))
        conc = tf.concat([conc_d, rgb], axis=-1)
        w_e = [consts.dim_w//2]
        c_e = [16]
        bottle_neck_w = 4

        while w_e[-1] > bottle_neck_w:
            w_e.append(w_e[-1]//2)
            c_e.append(c_e[-1]*2)
        if print_fn is None:
            print_fn = print
        print_fn('-- Normal refiner 0 encoder layers\' width', w_e)
        print_fn('-- Normal refiner 0 encoder layers\' channel', c_e)
        layers = [conc]
        for c in c_e:
            with tf.variable_scope('nml_rf_e_%d' % (len(layers))):
                nin_shape = layers[-1].get_shape().as_list()
                net = slim.conv2d(layers[-1], c, [4, 4], 2, padding='SAME',
                                  weights_initializer=initializers.xavier_initializer(),
                                  weights_regularizer=None,
                                  rate=1, normalizer_fn=slim.batch_norm,
                                  activation_fn=tf.nn.leaky_relu, scope='conv0')
                print_fn('-- Normal refiner encoder 0 layer %d:'%len(layers), nin_shape, '-->', net.get_shape().as_list())
                layers.append(net)

        w_d = [w_e[-1]*2]
        c_d = [c_e[-1]//2]
        while w_d[-1] < consts.dim_w:
            w_d.append(w_d[-1]*2)
            c_d.append(c_d[-1]//2)
        print_fn('-- Normal refiner 0 decoder layers\' width', w_d)
        print_fn('-- Normal refiner 0 decoderlayers\' channel', c_d)

        for ci, c in enumerate(c_d):
            with tf.variable_scope('nml_rf_d_%d' % (len(layers))):
                nin_shape = layers[-1].get_shape().as_list()
                net = tf.image.resize_bilinear(layers[-1], (nin_shape[1]*2, nin_shape[2]*2))

                net = tf.concat([net, layers[len(w_e)-ci-1]], axis=-1)   # U-net structure

                net = slim.conv2d(net, c, [4, 4], 1, padding='SAME',
                                  weights_initializer=initializers.xavier_initializer(),
                                  weights_regularizer=None,
                                  rate=1, normalizer_fn=slim.batch_norm,
                                  activation_fn=tf.nn.leaky_relu, scope='conv0')
                print_fn('-- Normal refiner decoder layer %d:'%len(layers), nin_shape, '-->', net.get_shape().as_list())
                layers.append(net)

        with tf.variable_scope('nml_rf_d_out'):
            nin_shape = layers[-1].get_shape().as_list()
            net = slim.conv2d(layers[-1], 3, [1, 1], 1, padding='SAME',
                              weights_initializer=initializers.xavier_initializer(),
                              weights_regularizer=None,
                              rate=1, normalizer_fn=None,
                              activation_fn=tf.nn.tanh, scope='conv0')       # output to (-1, 1)
            net = net + conc_d
            print_fn('-- Normal refiner 0 decoder layer %d:' % len(layers), nin_shape, '-->', net.get_shape().as_list())
            layers.append(net)

        return layers

    @staticmethod
    def _build_normal_refiner2(normal_1, normal_2, normal_3, print_fn=None):
        def build_u_net(normal, reuse, print_fn=None):
            conc = tf.image.resize_bilinear(normal, (consts.dim_h * 2, consts.dim_w * 2))
            w_e = [consts.dim_w // 2]
            c_e = [16]
            bottle_neck_w = 4

            while w_e[-1] > bottle_neck_w:
                w_e.append(w_e[-1] // 2)
                c_e.append(c_e[-1] * 2)
            if print_fn is None:
                print_fn = print
            if not reuse:
                print_fn('-- Normal refiner 1 encoder layers\' width', w_e)
                print_fn('-- Normal refiner 1 encoder layers\' channel', c_e)
            layers = [conc]
            for c in c_e:
                with tf.variable_scope('nml_rf2_e_%d' % (len(layers)), reuse=reuse):
                    nin_shape = layers[-1].get_shape().as_list()
                    net = slim.conv2d(layers[-1], c, [4, 4], 2, padding='SAME',
                                      weights_initializer=initializers.xavier_initializer(),
                                      weights_regularizer=None,
                                      rate=1, normalizer_fn=slim.batch_norm,
                                      activation_fn=tf.nn.leaky_relu, scope='conv0')
                    if not reuse:
                        print_fn('-- Normal refiner 1 encoder layer %d:' % len(layers), nin_shape, '-->',
                                 net.get_shape().as_list())
                    layers.append(net)

            w_d = [w_e[-1] * 2]
            c_d = [c_e[-1] // 2]
            while w_d[-1] < consts.dim_w:
                w_d.append(w_d[-1] * 2)
                c_d.append(c_d[-1] // 2)
            if not reuse:
                print_fn('-- Normal refiner 1 decoder layers\' width', w_d)
                print_fn('-- Normal refiner 1 decoderlayers\' channel', c_d)

            for ci, c in enumerate(c_d):
                with tf.variable_scope('nml_rf2_d_%d' % (len(layers)), reuse=reuse):
                    nin_shape = layers[-1].get_shape().as_list()
                    net = tf.image.resize_bilinear(layers[-1], (nin_shape[1] * 2, nin_shape[2] * 2))

                    net = tf.concat([net, layers[len(w_e) - ci - 1]], axis=-1)  # U-net structure

                    net = slim.conv2d(net, c, [4, 4], 1, padding='SAME',
                                      weights_initializer=initializers.xavier_initializer(),
                                      weights_regularizer=None,
                                      rate=1, normalizer_fn=slim.batch_norm,
                                      activation_fn=tf.nn.leaky_relu, scope='conv0')
                    if not reuse:
                        print_fn('-- Normal refiner 1 decoder layer %d:' % len(layers), nin_shape, '-->',
                                 net.get_shape().as_list())
                    layers.append(net)

            with tf.variable_scope('nml_rf2_d_out', reuse=reuse):
                nin_shape = layers[-1].get_shape().as_list()
                net = slim.conv2d(layers[-1], 3, [1, 1], 1, padding='SAME',
                                  weights_initializer=initializers.xavier_initializer(),
                                  weights_regularizer=None,
                                  rate=1, normalizer_fn=None,
                                  activation_fn=tf.nn.tanh, scope='conv0')  # output to (-1, 1)
                net = net + conc
                if not reuse:
                    print_fn('-- Normal refiner 1 decoder layer %d:' % len(layers), nin_shape, '-->',
                             net.get_shape().as_list())
                layers.append(net)

            return layers

        with tf.name_scope('normal_1_R'):
            normal_1_r = build_u_net(normal_1, reuse=False, print_fn=print_fn)
        with tf.name_scope('normal_2_R'):
            normal_2_r = build_u_net(normal_2, reuse=True, print_fn=print_fn)
        with tf.name_scope('normal_3_R'):
            normal_3_r = build_u_net(normal_3, reuse=True, print_fn=print_fn)
        return normal_1_r, normal_2_r, normal_3_r

    @staticmethod
    def _build_normal_discriminator(d_pred, d_gt, mask_fv_gt, mask_sv_gt, in_img, print_fn=None):
        if print_fn is None:
            print_fn = print

        m_conc = tf.concat([mask_fv_gt, mask_fv_gt, mask_fv_gt,
                            mask_sv_gt, mask_sv_gt, mask_sv_gt,
                            mask_fv_gt, mask_fv_gt, mask_fv_gt,
                            mask_sv_gt, mask_sv_gt, mask_sv_gt], axis=-1)
        d_pred_m = m_conc * d_pred      # mask out background
        d_gt_m = m_conc * d_gt          # mask out background
        conc_pred = tf.concat([d_pred_m, in_img], axis=-1)
        conc_gt = tf.concat([d_gt_m, in_img], axis=-1)
        conc_pred = conc_pred
        conc_gt = conc_gt

        def build_D(conc, reuse=False):
            batch_sz = conc.get_shape().as_list()[0]
            layer_w = conc.get_shape().as_list()[2]
            w_e = [layer_w // 2]
            c_e = [16]
            while w_e[-1] > 16:
                w_e.append(w_e[-1] // 2)
                c_e.append(min(c_e[-1] * 2, 64))
            if not reuse:
                print_fn('-- Normal discriminator encoder layers\' width', w_e)
                print_fn('-- Normal discriminator encoder layers\' channel', c_e)

            layers = [conc]
            for c in c_e:
                with tf.variable_scope('nml_dis_e_%d' % (len(layers)), reuse=reuse):
                    nin_shape = layers[-1].get_shape().as_list()
                    net = slim.conv2d(layers[-1], c, [3, 3], 1, padding='SAME',
                                      weights_initializer=initializers.xavier_initializer(),
                                      weights_regularizer=None,
                                      rate=1, normalizer_fn=slim.batch_norm,
                                      activation_fn=tf.nn.leaky_relu, scope='conv0')
                    net = slim.max_pool2d(net, [2, 2], [2, 2],  padding='SAME', scope='maxp0')
                    if not reuse:
                        print_fn('-- Normal discriminator encoder layer %d:'%len(layers), nin_shape, '-->', net.get_shape().as_list())
                    layers.append(net)
            with tf.variable_scope('nml_dis_out', reuse=reuse):
                nin_shape = layers[-1].get_shape().as_list()
                net = slim.conv2d(layers[-1], 1, [1, 1], 1, padding='SAME',
                                  weights_initializer=initializers.xavier_initializer(),
                                  weights_regularizer=None,
                                  rate=1, normalizer_fn=None,
                                  activation_fn=tf.nn.sigmoid, scope='conv0')
                if not reuse:
                    print_fn('-- Normal discriminator encoder layer %d:' % len(layers), nin_shape, '-->', net.get_shape().as_list())
                layers.append(net)
            return layers

        with tf.name_scope('Dis_real'):
            d_out_gt = build_D(tf.concat([conc_gt[:, :, :, 0:3], conc_gt[:, :, :, 12:15]], axis=-1), reuse=False)
        with tf.name_scope('Dis_fake'):
            d_out_pred = build_D(tf.concat([conc_pred[:, :, :, 0:3], conc_pred[:, :, :, 12:15]], axis=-1), reuse=True)

        # with tf.name_scope('Dis_real_0'):
        #     d_out_gt0 = build_D(conc_gt[:, :, :, 0:3], reuse=False)
        # with tf.name_scope('Dis_real_1'):
        #     d_out_gt1 = build_D(conc_gt[:, :, :, 3:6], reuse=True)
        # with tf.name_scope('Dis_real_2'):
        #     d_out_gt2 = build_D(conc_gt[:, :, :, 6:9], reuse=True)
        # with tf.name_scope('Dis_real_3'):
        #     d_out_gt3 = build_D(conc_gt[:, :, :, 9:12], reuse=True)
        # with tf.name_scope('Dis_fake_0'):
        #     d_out_pred0 = build_D(conc_pred[:, :, :, 0:3], reuse=True)
        # with tf.name_scope('Dis_fake_1'):
        #     d_out_pred1 = build_D(conc_pred[:, :, :, 3:6], reuse=True)
        # with tf.name_scope('Dis_fake_2'):
        #     d_out_pred2 = build_D(conc_pred[:, :, :, 6:9], reuse=True)
        # with tf.name_scope('Dis_fake_3'):
        #     d_out_pred3 = build_D(conc_pred[:, :, :, 9:12], reuse=True)

        # d_out_gt = tf.concat([d_out_gt0[-1], d_out_gt1[-1], d_out_gt2[-1], d_out_gt3[-1]], axis=-1)
        # d_out_pred = tf.concat([d_out_pred0[-1], d_out_pred1[-1], d_out_pred2[-1], d_out_pred3[-1]], axis=-1)
        return d_out_gt[-1], d_out_pred[-1]

    @staticmethod
    def _build_loss(vol_pred, vol_gt,
                    mask_fv_gt, mask_sv_gt,
                    normal_hd_gt, normal_hd_pred,
                    dis_real, dis_fake,
                    lamb_sil=0.1, lamb_nml_rf=0.01, lamb_dis=0.001,
                    w=0.7):
        log('Constructing loss function...')

        s = 1000    # to scale the loss
        shp = mask_fv_gt.get_shape().as_list()
        with tf.name_scope('loss'):
            # volume loss
            vol_loss = s * tf.reduce_mean(-w * tf.reduce_mean(vol_gt * tf.log(vol_pred + 1e-8))
                                          - (1 - w) * tf.reduce_mean((1 - vol_gt) * tf.log(1 - vol_pred + 1e-8)))
            # silhouette loss
            mask_fv_pred, mask_sv_pred = Trainer._build_sil_projector(vol_pred)
            #mask_fv_gt_p, mask_sv_gt_p = Trainer._build_sil_projector(vol_gt)
            mask_fv_gt_rs = tf.image.resize_bilinear(mask_fv_gt, (shp[1]//2, shp[2]//2))
            mask_sv_gt_rs = tf.image.resize_bilinear(mask_sv_gt, (shp[1]//2, shp[2]//2))
            sil_loss_fv = s * tf.reduce_mean(-tf.reduce_mean(mask_fv_gt_rs * tf.log(mask_fv_pred + 1e-8))
                                             -tf.reduce_mean((1-mask_fv_gt_rs) * tf.log(1 - mask_fv_pred + 1e-8)))
            sil_loss_sv = s * tf.reduce_mean(-tf.reduce_mean(mask_sv_gt_rs * tf.log(mask_sv_pred + 1e-8))
                                             -tf.reduce_mean((1-mask_sv_gt_rs) * tf.log(1 - mask_sv_pred + 1e-8)))
            sil_loss = sil_loss_fv + sil_loss_sv

            # normal refinement loss
            normal_loss = 0
            for i in range(4):
                normal_hd_gt_ = normal_hd_gt[:, :, :, (i*3):(i*3+3)]
                normal_hd_pred_ = normal_hd_pred[:, :, :, (i*3):(i*3+3)]
                normal_cos = 1 - tf.reduce_sum(normal_hd_gt_*normal_hd_pred_, axis=-1, keepdims=True) \
                             / (tf.norm(normal_hd_gt_, axis=-1, keepdims=True)*tf.norm(normal_hd_pred_, axis=-1, keepdims=True))
                # mask out invalid areas
                if i % 2 == 0:
                    normal_loss += s * tf.reduce_mean(mask_fv_gt*normal_cos)
                    normal_loss += s * 0.001 * tf.reduce_mean(mask_fv_gt*tf.square(normal_hd_pred_-normal_hd_gt_))
                else:
                    normal_loss += s * tf.reduce_mean(mask_sv_gt*normal_cos)
                    normal_loss += s * 0.001 * tf.reduce_mean(mask_sv_gt*tf.square(normal_hd_pred_-normal_hd_gt_))

            # normal discriminator loss
            dis_d_real_loss = s * tf.reduce_mean(tf.square(dis_fake))
            dis_d_fake_loss = s * tf.reduce_mean(tf.square(1-dis_real))
            dis_d_loss = dis_d_real_loss + dis_d_fake_loss
            dis_g_loss = s * tf.reduce_mean(tf.square(1-dis_fake))

            # total loss
            recon_loss = vol_loss + lamb_sil * sil_loss                     # reconstruction loss
            nr_loss = lamb_nml_rf * normal_loss + lamb_dis * dis_g_loss     # normal refinement loss
            total_loss = recon_loss + nr_loss                               # total loss

            loss_collection = {}
            loss_collection['vol_loss'] = vol_loss
            loss_collection['sil_loss'] = sil_loss
            loss_collection['normal_loss'] = normal_loss
            loss_collection['dis_d_real_loss'] = dis_d_real_loss
            loss_collection['dis_d_fake_loss'] = dis_d_fake_loss
            loss_collection['dis_d_loss'] = dis_d_loss
            loss_collection['dis_g_loss'] = dis_g_loss
            loss_collection['recon_loss'] = recon_loss
            loss_collection['nr_loss'] = nr_loss
            loss_collection['total_loss'] = total_loss
        return loss_collection

    @staticmethod
    def _build_optimizer(lr, recon_loss, nr_loss, total_loss, dis_loss):
        log('Constructing optimizer...')

        recon_var_list = [var for var in tf.trainable_variables() if not var.name.startswith('nml_rf') and not var.name.startswith('nml_dis')]
        nr_var_list = [var for var in tf.trainable_variables() if var.name.startswith('nml_rf') and not var.name.startswith('nml_dis')]
        all_var_list = [var for var in tf.trainable_variables() if not var.name.startswith('nml_dis')]
        dis_var_list = [var for var in tf.trainable_variables() if var.name.startswith('nml_dis')]

        with tf.name_scope('recon_optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                recon_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(recon_loss, var_list=recon_var_list)
        with tf.name_scope('nml_rf_optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                dr_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(nr_loss, var_list=nr_var_list)
        with tf.name_scope('all_optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                all_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss, var_list=all_var_list)
        with tf.name_scope('dis_optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                dis_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(dis_loss, var_list=dis_var_list)
        return recon_opt, dr_opt, all_opt, dis_opt

    @staticmethod
    def _setup_summary(sess, graph_dir, loss_collection):
        loss_scalar_s = []
        for lk in loss_collection:
            loss_s = tf.summary.scalar('loss/%s' % (lk), loss_collection[lk])
            loss_scalar_s.append(loss_s)

        merged_scalar_loss = tf.summary.merge([loss_s for loss_s in loss_scalar_s])
        writer = tf.summary.FileWriter(graph_dir, sess.graph)
        return merged_scalar_loss, writer

    def _setup_saver(self, pre_model_dir):
        # load pre-trained model to fine-tune or resume training
        log('Constructing saver...')
        if pre_model_dir is not None:
            ckpt_prev = tf.train.get_checkpoint_state(pre_model_dir)
            if ckpt_prev:
                saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables()])
                saver.restore(self.sess, ckpt_prev.model_checkpoint_path)
                logger.write('Loaded model %s' % pre_model_dir)
            else:
                logger.write('Unable to load the pretrained model. ')
        saver = tf.train.Saver(max_to_keep=1000)
        return saver

    @staticmethod
    def _save_tuple(conc_imgs, smpl_v_volumes, mesh_volumes, dir, idx):
        batch_sz = conc_imgs.shape[0]
        for bi in range(batch_sz):
            cv.imwrite('%s/color_%d.png' % (dir, batch_sz * idx + bi), cv.cvtColor(np.uint8(conc_imgs[bi, :, :, 0:3] * 255), cv.COLOR_BGRA2RGB))
            cv.imwrite('%s/vmap_%d.png' % (dir, batch_sz * idx + bi), np.uint8(conc_imgs[bi, :, :, 3:6] * 255))
            cv.imwrite('%s/mask_%d.png' % (dir, batch_sz * idx + bi), np.uint8(conc_imgs[bi, :, :, 6] * 255))
            cv.imwrite('%s/normal_%d.png' % (dir, batch_sz * idx + bi), np.uint16(conc_imgs[bi, :, :, 10:13] * 32767.5 + 32767.5))

    @staticmethod
    def _save_results_raw_training(mesh_volume, refined_normal, orig_normal, test_dir, idx):
        batch_sz = mesh_volume.shape[0]
        for bi in range(batch_sz):
            sio.savemat('%s/mesh_volume_%d.obj' % (test_dir, batch_sz*idx+bi),
                        {'mesh_volume': mesh_volume[bi, :, :, :, 0]}, do_compression=False)

        for bi in range(batch_sz):
            for vi in range(4):
                refined_normal_ = refined_normal[bi, :, :, (3*vi):(3*vi+3)]
                refined_normal_l = np.sqrt(refined_normal_[:, :, 0] * refined_normal_[:, :, 0]+
                                           refined_normal_[:, :, 1] * refined_normal_[:, :, 1] +
                                           refined_normal_[:, :, 2] * refined_normal_[:, :, 2])
                refined_normal_ /= np.expand_dims(refined_normal_l, axis=-1)
                refined_normal[bi, :, :, (3 * vi):(3 * vi + 3)] = refined_normal_

                original_normal_ = orig_normal[bi, :, :, (3*vi):(3*vi+3)]
                original_normal_l = np.sqrt(original_normal_[:, :, 0] * original_normal_[:, :, 0] +
                                            original_normal_[:, :, 1] * original_normal_[:, :, 1] +
                                            original_normal_[:, :, 2] * original_normal_[:, :, 2])
                original_normal_ /= np.expand_dims(original_normal_l, axis=-1)
                orig_normal[bi, :, :, (3 * vi):(3 * vi + 3)] = original_normal_

            cv.imwrite('%s/normal_0_%d.png' % (test_dir, batch_sz * idx + bi), np.uint16(refined_normal[bi, :, :, 0:3] * 32767.5 + 32767.5))
            cv.imwrite('%s/normal_1_%d.png' % (test_dir, batch_sz * idx + bi), np.uint16(refined_normal[bi, :, :, 3:6] * 32767.5 + 32767.5))
            cv.imwrite('%s/normal_2_%d.png' % (test_dir, batch_sz * idx + bi), np.uint16(refined_normal[bi, :, :, 6:9] * 32767.5 + 32767.5))
            cv.imwrite('%s/normal_3_%d.png' % (test_dir, batch_sz * idx + bi), np.uint16(refined_normal[bi, :, :, 9:12] * 32767.5 + 32767.5))
            
            cv.imwrite('%s/normal_0_%d_.png' % (test_dir, batch_sz * idx + bi), np.uint16(orig_normal[bi, :, :, 0:3] * 32767.5 + 32767.5))
            cv.imwrite('%s/normal_1_%d_.png' % (test_dir, batch_sz * idx + bi), np.uint16(orig_normal[bi, :, :, 3:6] * 32767.5 + 32767.5))
            cv.imwrite('%s/normal_2_%d_.png' % (test_dir, batch_sz * idx + bi), np.uint16(orig_normal[bi, :, :, 6:9] * 32767.5 + 32767.5))
            cv.imwrite('%s/normal_3_%d_.png' % (test_dir, batch_sz * idx + bi), np.uint16(orig_normal[bi, :, :, 9:12] * 32767.5 + 32767.5))

    @staticmethod
    def _save_results_raw_testing(mesh_volume, refined_normal, orig_normal, test_dir, prefix):
        batch_sz = mesh_volume.shape[0]
        assert batch_sz == 1        # only use for testing
        # mesh_volume = np.squeeze(mesh_volume)
        for bi in range(batch_sz):
            sio.savemat('%s/%s_volume_out.mat' % (test_dir, prefix),
                        {'mesh_volume': mesh_volume[bi, :, :, :, 0]}, do_compression=False)

        for bi in range(batch_sz):
            for vi in range(4):
                refined_normal_ = refined_normal[bi, :, :, (3*vi):(3*vi+3)]
                refined_normal_l = np.sqrt(refined_normal_[:, :, 0] * refined_normal_[:, :, 0]+
                                           refined_normal_[:, :, 1] * refined_normal_[:, :, 1] +
                                           refined_normal_[:, :, 2] * refined_normal_[:, :, 2])
                refined_normal_ /= np.expand_dims(refined_normal_l, axis=-1)

                original_normal_ = orig_normal[bi, :, :, (3*vi):(3*vi+3)]
                original_normal_l = np.sqrt(original_normal_[:, :, 0] * original_normal_[:, :, 0] +
                                            original_normal_[:, :, 1] * original_normal_[:, :, 1] +
                                            original_normal_[:, :, 2] * original_normal_[:, :, 2])
                original_normal_ /= np.expand_dims(original_normal_l, axis=-1)

                cv.imwrite('%s/%s_normal_%d.png' % (test_dir, prefix, vi), np.uint16(refined_normal_ * 32767.5 + 32767.5))
                cv.imwrite('%s/%s_normal_orig_%d.png' % (test_dir, prefix, vi), np.uint16(original_normal_ * 32767.5 + 32767.5))
