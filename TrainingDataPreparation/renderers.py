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
from MyRenderer import ColoredRenderer
from MyCamera import ProjectPointsOrthogonal

import util


def _project_vertices(v, w, h, cam_r, cam_t):
    """projects vertices onto image plane"""
    V = ch.array(v)
    U = ProjectPointsOrthogonal(v=V, f=[w, w], c=[w/2., h/2.],
                      k=ch.zeros(5), t=cam_t, rt=cam_r)
    return U


def _render_color_model_with_lighting(w, h, v, vn, vc, f, u,
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
                         frustum={'width': w, 'height': h, 'near': 1.0, 'far': 20})
    return rn.r


def _render_color_model_without_lighting(w, h, v, vc, f, u,
                                         bg_img=None):
    """renders colored model without lighting effect"""
    V = ch.array(v)
    A = vc
    black_img = np.array(np.zeros((w, h, 3)), dtype=np.float32)
    bg_img_ = bg_img if bg_img is not None else black_img
    rn = ColoredRenderer(camera=u, v=V, f=f, vc=A, background_image=bg_img_,
                         frustum={'width': w, 'height': h, 'near': 1.0, 'far': 20})
    return rn.r


def _render_mask(w, h, v, f, u):
    """renders silhouette"""
    V = ch.array(v)
    A = np.ones(v.shape)
    rn = ColoredRenderer(camera=u, v=V, f=f, vc=A, bgcolor=ch.zeros(3),
                         frustum={'width': w, 'height': h, 'near': 1.0, 'far': 20})
    return rn.r


def render_training_pairs(mesh, smpl, img_w, img_h, camera_r, camera_t, color_bg,
                          sh_comps=None, light_c=ch.ones(3),
                          vlight_pos=None, vlight_color=None):
    """generates training image pairs
    Will generate color image, mask, semantic map, normal map
    """
    v_, v_smpl_ = mesh['v'], smpl['v']

    # render color image
    # To avoid aliasing, I render the image with 2x resolution and then resize it
    # See: https://stackoverflow.com/questions/22069167/opencv-how-to-smoothen-boundary
    u = _project_vertices(v_, img_w*2, img_h*2, camera_r, camera_t)
    color_bg = cv.resize(color_bg, (img_w*2, img_h*2))
    img = _render_color_model_with_lighting(img_w*2, img_h*2, v_, mesh['vn'],
                                            mesh['vc'], mesh['f'], u,
                                            sh_comps=sh_comps, light_c=light_c,
                                            vlight_pos=vlight_pos,
                                            vlight_color=vlight_color,
                                            bg_img=color_bg)
    img = cv.resize(img, (img_w, img_h))
    img = np.float32(np.copy(img))

    # render silhouette
    u = _project_vertices(v_, img_w, img_h, camera_r, camera_t)
    msk = _render_mask(img_w, img_h, v_, mesh['f'], u)
    msk = np.float32(np.copy(msk))

    # render normal maps
    if 'n' in mesh:
        n_ = mesh['n']*0.5 + 0.5
    else:
        vn = util.calc_normal(mesh)
        n_ = vn*0.5 + 0.5
    nml = _render_color_model_without_lighting(img_w, img_h, v_, n_, mesh['f'],
                                               u, bg_img=None)
    nml = np.float32(np.copy(nml))

    # render semantic map
    u = _project_vertices(v_smpl_, img_w, img_h, camera_r, camera_t)
    vc_smpl = util.get_smpl_semantic_code()
    smap = _render_color_model_without_lighting(img_w, img_h, v_smpl_, vc_smpl,
                                                smpl['f'], u, bg_img=None)
    smap = np.float32(np.copy(smap))

    return img, msk, nml, smap
