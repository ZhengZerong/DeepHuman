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

"""Rendering APIs"""

from __future__ import division, print_function

import numpy as np
import chumpy as ch
import cv2 as cv

from opendr.lighting import LambertianPointLight, SphericalHarmonics
from opendr.renderer import ColoredRenderer, TexturedRenderer, DepthRenderer
from opendr.camera import ProjectPoints

import CommonUtil as util


def project_vertices(v, w, h, cam_r, cam_t):
    """projects vertices onto image plane"""
    V = ch.array(v)
    U = ProjectPoints(v=V, f=[w, w], c=[w/2., h/2.],
                      k=ch.zeros(5), t=cam_t, rt=cam_r)
    return U


def render_color_model_with_lighting(w, h, v, vn, vc, f, u,
                                      sh_comps=None, light_c=ch.ones(3),
                                      vlight_pos=None, vlight_color=None,
                                      bg_img=None):
    """renders colored model with lighting effect"""
    assert(sh_comps is not None or vlight_pos is not None)
    V = ch.array(v)
    A = np.zeros_like(v)

    # SH lighting
    if sh_comps is not None:
        A += vc * SphericalHarmonics(vn=vn, components=sh_comps, light_color=light_c)

    # single point lighting (grey light)
    if vlight_color is not None and vlight_pos is not None \
            and len(vlight_pos.shape) == 1:
        A += LambertianPointLight(f=f, v=v, num_verts=len(v), light_pos=vlight_pos,
                                  light_color=vlight_color, vc=vc)

    # multiple point lighting (grey light)
    if vlight_color is not None and vlight_pos is not None \
            and len(vlight_pos.shape) == 2:
        for vlp in vlight_pos:
            A += LambertianPointLight(f=f, v=v, num_verts=len(v), light_pos=vlp,
                                      light_color=vlight_color, vc=vc)

    black_img = np.array(np.zeros((w, h, 3)), dtype=np.float32)
    bg_img_ = bg_img if bg_img is not None else black_img

    rn = ColoredRenderer(camera=u, v=V, f=f, vc=A, background_image=bg_img_,
                         frustum={'width': w, 'height': h, 'near': 0.1, 'far': 20})
    return rn.r


def render_color_model_without_lighting(w, h, v, vc, f, u,
                                         bg_img=None):
    """renders colored model without lighting effect"""
    V = ch.array(v)
    A = vc
    black_img = np.array(np.zeros((w, h, 3)), dtype=np.float32)
    bg_img_ = bg_img if bg_img is not None else black_img
    rn = ColoredRenderer(camera=u, v=V, f=f, vc=A, background_image=bg_img_,
                         frustum={'width': w, 'height': h, 'near': 0.1, 'far': 20})
    return rn.r


def render_mask(w, h, v, f, u):
    """renders silhouette"""
    V = ch.array(v)
    A = np.ones(v.shape)
    rn = ColoredRenderer(camera=u, v=V, f=f, vc=A, bgcolor=ch.zeros(3),
                         frustum={'width': w, 'height': h, 'near': 0.1, 'far': 20})
    return rn.r


def compress_along_z_axis(v, v_smpl, z1=-1.01, z2=-1.0):
    """rescales model along z axis to approximate orthogonal projection"""
    min_z = np.min(v[:, 2])
    max_z = np.max(v[:, 2])
    v_ = np.copy(v)
    v_smpl_ = np.copy(v_smpl)
    v_[:, 2] = (v_[:, 2] - min_z) / (max_z - min_z) * (z2 - z1) + z1
    v_smpl_[:, 2] = (v_smpl_[:, 2] - min_z) / (max_z - min_z) * (z2 - z1) + z1
    return v_, v_smpl_


def compress_along_z_axis_single(v, z1=-1.01, z2=-1.0):
    """rescales model along z axis to approximate orthogonal projection"""
    min_z = np.min(v[:, 2])
    max_z = np.max(v[:, 2])
    v_ = np.copy(v)
    v_[:, 2] = (v_[:, 2] - min_z) / (max_z - min_z) * (z2 - z1) + z1
    return v_
