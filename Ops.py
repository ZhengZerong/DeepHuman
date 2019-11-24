from __future__ import absolute_import, print_function, division
"""
    TF operations
    My encapsulates of common network operations
    Modified from: https://github.com/Yang7879/3D-RecGAN-extended/blob/master/tools.py
"""

import tensorflow as tf
import numpy as np
import math

class Ops:

    @staticmethod
    def featrue_affine(f_3d, f_2d_a, f_2d_b):
        shape_2d_a = f_2d_a.get_shape().as_list()     # BHWC
        shape_2d_b = f_2d_b.get_shape().as_list()     # BHWC
        shape_3d = f_3d.get_shape().as_list()     # BDHWC

        assert shape_2d_b == shape_2d_a
        assert shape_2d_a[1] == shape_3d[2] and shape_2d_a[2] == shape_3d[3] and shape_2d_a[3] == shape_3d[4]

        f_2d_a = tf.reshape(f_2d_a, (shape_2d_a[0], 1, shape_2d_a[1], shape_2d_a[2], shape_2d_a[3]))
        f_2d_b = tf.reshape(f_2d_b, (shape_2d_a[0], 1, shape_2d_a[1], shape_2d_a[2], shape_2d_a[3]))

        f_3d = f_3d * f_2d_a + f_2d_b
        return f_3d

    @staticmethod
    def lrelu(x, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    @staticmethod
    def relu(x):
        return tf.nn.relu(x)

    @staticmethod
    def xxlu(x,label,name=None):
        if label =='relu':
            return  Ops.relu(x)
        if label =='lrelu':
            return  Ops.lrelu(x,leak=0.2)

    @staticmethod
    def variable_statistics(var, name):
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    @staticmethod
    def fc(x, out_d, name):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        in_d = x.get_shape()[1]
        w = tf.get_variable(name + '_w', [in_d, out_d], initializer=xavier_init)
        b = tf.get_variable(name + '_b', [out_d], initializer=zero_init)
        y = tf.nn.bias_add(tf.matmul(x, w), b)
        Ops.variable_statistics(w, name)
        return y

    @staticmethod
    def maxpool3d(x,k,s,pad='SAME'):
        ker =[1,k,k,k,1]
        str =[1,s,s,s,1]
        y = tf.nn.max_pool3d(x,ksize=ker,strides=str,padding=pad)
        return y

    @staticmethod
    def conv3d(x, k, out_c, str, name, pad='SAME'):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        in_c = x.get_shape()[4]
        w = tf.get_variable(name + '_w', [k, k, k, in_c, out_c], initializer=xavier_init)
        b = tf.get_variable(name + '_b', [out_c], initializer=zero_init)
        stride = [1, str, str, str, 1]
        y = tf.nn.bias_add(tf.nn.conv3d(x, w, stride, pad), b)
        Ops.variable_statistics(w, name)
        return y

    @staticmethod
    def deconv3d(x, k, out_c, str, name, pad='SAME'):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        [_, in_d1, in_d2, in_d3, in_c] = x.get_shape()
        in_d1 = int(in_d1); in_d2 = int(in_d2); in_d3 = int(in_d3); in_c = int(in_c)
        bat = tf.shape(x)[0]
        w = tf.get_variable(name + '_w', [k, k, k, out_c, in_c], initializer=xavier_init)
        b = tf.get_variable(name + '_b', [out_c], initializer=zero_init)
        out_shape = [bat, in_d1 * str, in_d2 * str, in_d3 * str, out_c]
        stride = [1, str, str, str, 1]
        y = tf.nn.conv3d_transpose(x, w, output_shape=out_shape, strides=stride, padding=pad)
        y = tf.nn.bias_add(y, b)
        Ops.variable_statistics(w, name)
        return y

    @staticmethod
    def get_variable_num(logger=None):
        cout_fn = print if logger is None else logger.write

        from functools import reduce
        from operator import mul
        num_params = 0
        cout_fn('All trainable variable: ')
        for variable in tf.trainable_variables():
            cout_fn('-- ', variable.name)
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params

    @staticmethod
    def compute_l1_error(real, fake):
        assert real.get_shape().as_list() == fake.get_shape().as_list()
        return tf.reduce_mean(tf.abs(real-fake))

    @staticmethod
    def compute_l2_error(real, fake):
        assert real.get_shape().as_list() == fake.get_shape().as_list()
        return tf.reduce_mean(tf.square(real - fake))