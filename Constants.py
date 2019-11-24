from __future__ import division, absolute_import, print_function


class Constants:
    def __init__(self):
        self.dim_w = 128
        self.dim_h = 192
        self.hb_ratio = self.dim_h/self.dim_w
        self.real_h = 1.0
        self.real_w = self.real_h /self.dim_h * self.dim_w
        self.voxel_size = self.real_h/self.dim_h
        self.tau = 0.5
        self.K = 100
        self.fill = True

        # loss weights
        self.lamb_sil = 0.02
        self.lamb_dis = 0.001
        self.lamb_nml_rf = 0.1


consts = Constants()