#!/usr/bin/env python
# encoding: utf-8
"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

__all__ = ['ProjectPoints3D', 'ProjectPoints', 'RigidTransform']

import chumpy as ch
from chumpy import depends_on, Ch
from opendr.cvwrap import cv2
import numpy as np
import scipy.sparse as sp
from chumpy.utils import row, col
from opendr.geometry import Rodrigues


def RigidTransformSlow(**kwargs):
    # Returns a Ch object with dterms 'v', 'rt', and 't'

    result = Ch(lambda v, rt, t: v.dot(Rodrigues(rt=rt)) + t)
    if len(kwargs) > 0:
        result.set(**kwargs)
    return result


class RigidTransform(Ch):
    dterms = 'v', 'rt', 't'

    def compute_r(self):
        return (cv2.Rodrigues(self.rt.r)[0].dot(self.v.r.T) + col(self.t.r)).T.copy()

    def compute_dr_wrt(self, wrt):

        if wrt not in (self.v, self.rt, self.t):
            return

        if wrt is self.t:
            if not hasattr(self, '_drt') or self._drt.shape[0] != self.v.r.size:
                IS = np.arange(self.v.r.size)
                JS = IS % 3
                data = np.ones(len(IS))
                self._drt = sp.csc_matrix((data, (IS, JS)))
            return self._drt

        if wrt is self.rt:
            rot, rot_dr = cv2.Rodrigues(self.rt.r)
            rot_dr = rot_dr.reshape((3, 3, 3))
            dr = np.einsum('abc, zc -> zba', rot_dr, self.v.r).reshape((-1, 3))
            return dr

        if wrt is self.v:
            rot = cv2.Rodrigues(self.rt.r)[0]

            IS = np.repeat(np.arange(self.v.r.size), 3)
            JS = np.repeat(np.arange(self.v.r.size).reshape((-1, 3)), 3, axis=0)
            data = np.vstack([rot for i in range(self.v.r.size / 3)])
            result = sp.csc_matrix((data.ravel(), (IS.ravel(), JS.ravel())))
            return result


class ProjectPointsOrthogonal(Ch):
    dterms = 'v', 'rt', 't', 'f', 'c', 'k'

    def is_valid(self):
        if any([len(v.r.shape) > 1 for v in [self.rt, self.t, self.f, self.c, self.k]]):
            return False, 'rt, t, f, c, and k must be 1D'

        if any([v.r.size != 3 for v in [self.rt, self.t]]):
            return False, 'rt and t must have size=3'

        if any([v.r.size != 2 for v in [self.f, self.c]]):
            return False, 'f and c must have size=2'

        return True, ''

    def compute_r(self):
        return self.r_and_derivatives[0].squeeze()
        # return self.get_r_and_derivatives(self.v.r, self.rt.r, self.t.r, self.f.r, self.c.r, self.k.r)[0].squeeze()

    def compute_dr_wrt(self, wrt):
        if wrt not in [self.v, self.rt, self.t, self.f, self.c, self.k]:
            return None

        j = self.r_and_derivatives[1]
        if wrt is self.rt:
            return j[:, :3]
        elif wrt is self.t:
            return j[:, 3:6]
        elif wrt is self.f:
            return j[:, 6:8]
        elif wrt is self.c:
            return j[:, 8:10]
        elif wrt is self.k:
            return j[:, 10:10 + self.k.size]
        elif wrt is self.v:
            rot = cv2.Rodrigues(self.rt.r)[0]
            data = np.asarray(j[:, 3:6].dot(rot), order='C').ravel()
            IS = np.repeat(np.arange(self.v.r.size * 2 / 3), 3)
            JS = np.asarray(np.repeat(np.arange(self.v.r.size).reshape((-1, 3)), 2, axis=0),
                            order='C').ravel()
            result = sp.csc_matrix((data, (IS, JS)))
            return result

    # def unproject_points(self, uvd, camera_space=False):
    #     cam = ProjectPoints3D(**{k: getattr(self, k) for k in self.dterms if hasattr(self, k)})
    #
    #     try:
    #         xy_undistorted_camspace = cv2.undistortPoints(
    #             np.asarray(uvd[:, :2].reshape((1, -1, 2)).copy()), np.asarray(cam.camera_mtx),
    #             cam.k.r)
    #         xyz_camera_space = np.hstack((xy_undistorted_camspace.squeeze(), col(uvd[:, 2])))
    #         xyz_camera_space[:, :2] *= col(xyz_camera_space[:, 2])  # scale x,y by z
    #         if camera_space:
    #             return xyz_camera_space
    #         other_answer = xyz_camera_space - row(cam.view_mtx[:, 3])  # translate
    #         result = other_answer.dot(cam.view_mtx[:, :3])  # rotate
    #     except:  # slow way, probably not so good. But doesn't require cv2.undistortPoints.
    #         cam.v = np.ones_like(uvd)
    #         ch.minimize(cam - uvd, x0=[cam.v], method='dogleg', options={'disp': 0})
    #         result = cam.v.r
    #     return result
    #
    # def unproject_depth_image(self, depth_image, camera_space=False):
    #     us = np.arange(depth_image.size) % depth_image.shape[1]
    #     vs = np.arange(depth_image.size) // depth_image.shape[1]
    #     ds = depth_image.ravel()
    #     uvd = ch.array(np.vstack((us.ravel(), vs.ravel(), ds.ravel())).T)
    #     xyz = self.unproject_points(uvd, camera_space=camera_space)
    #     return xyz.reshape((depth_image.shape[0], depth_image.shape[1], -1))

    @depends_on('f', 'c')
    def camera_mtx(self):
        return np.array(
            [[self.f.r[0], 0, self.c.r[0]], [0., self.f.r[1], self.c.r[1]], [0., 0., 0.]],
            dtype=np.float64)

    @depends_on('t', 'rt')
    def view_mtx(self):
        R = cv2.Rodrigues(self.rt.r)[0]
        return np.hstack((R, col(self.t.r)))

    @depends_on('v', 'rt', 't', 'f', 'c', 'k')
    def r_and_derivatives(self):
        v = self.v.r.reshape((-1, 3)).copy()
        v_proj = np.matmul(v, self.view_mtx[:3, :3].transpose()) + self.view_mtx[:3, 3]
        v_proj = v_proj[:, :2]
        v_proj = v_proj * self.f.r[np.newaxis, :] + self.c.r[np.newaxis, :]
        J = np.zeros((self.v.r.shape[0], 10 + self.k.size))
        return v_proj, J
        #
        # v = self.v.r.reshape((-1, 3)).copy()
        # return cv2.projectPoints(v, self.rt.r, self.t.r, self.camera_mtx, self.k.r)

    @property
    def view_matrix(self):
        R = cv2.Rodrigues(self.rt.r)[0]
        return np.hstack((R, col(self.t.r)))


# class ProjectPoints3D(ProjectPoints):
#     dterms = 'v', 'rt', 't', 'f', 'c', 'k'
#
#     def compute_r(self):
#         result = ProjectPoints.compute_r(self)
#         return np.hstack((result, col(self.z_coords.r)))
#
#     @property
#     def z_coords(self):
#         assert (self.v.r.shape[1] == 3)
#         return RigidTransform(v=self.v, rt=self.rt, t=self.t)[:, 2]
#
#     def compute_dr_wrt(self, wrt):
#         result = ProjectPoints.compute_dr_wrt(self, wrt)
#         if result is None:
#             return None
#
#         if sp.issparse(result):
#             drz = self.z_coords.dr_wrt(wrt).tocoo()
#             result = result.tocoo()
#             result.row = result.row * 3 / 2
#
#             IS = np.concatenate((result.row, drz.row * 3 + 2))
#             JS = np.concatenate((result.col, drz.col))
#             data = np.concatenate((result.data, drz.data))
#
#             result = sp.csc_matrix((data, (IS, JS)), shape=(self.v.r.size, wrt.r.size))
#         else:
#             bigger = np.zeros((result.shape[0] / 2, 3, result.shape[1]))
#             bigger[:, :2, :] = result.reshape((-1, 2, result.shape[-1]))
#             drz = self.z_coords.dr_wrt(wrt)
#             if drz is not None:
#                 if sp.issparse(drz):
#                     drz = drz.todense()
#                 bigger[:, 2, :] = drz.reshape(bigger[:, 2, :].shape)
#
#             result = bigger.reshape((-1, bigger.shape[-1]))
#
#         return result
#
#
# def main():
#     import unittest
#     from test_camera import TestCamera
#     suite = unittest.TestLoader().loadTestsFromTestCase(TestCamera)
#     unittest.TextTestRunner(verbosity=2).run(suite)
#
#
# if __name__ == '__main__':
#     main()
#
