import numpy as np
import cv2 as cv

def load_obj_data(filename):
    """load model data from .obj file"""
    v_list = []  # vertex coordinate
    vt_list = []  # vertex texture coordinate
    vc_list = []  # vertex color
    vn_list = []  # vertex normal
    f_list = []  # face vertex indices
    fn_list = []  # face normal indices
    ft_list = []  # face texture indices

    # read data
    fp = open(filename, 'r')
    lines = fp.readlines()
    fp.close()

    for line in lines:
        line_data = line.strip().split(' ')

        # parse vertex cocordinate
        if line_data[0] == 'v':
            v_list.append((float(line_data[1]), float(line_data[2]), float(line_data[3])))
            if len(line_data) == 7:
                vc_list.append((float(line_data[4]), float(line_data[5]), float(line_data[6])))
            else:
                vc_list.append((0.5, 0.5, 0.5))

        # parse vertex texture coordinate
        if line_data[0] == 'vt':
            vt_list.append((float(line_data[1]), float(line_data[2])))

        # parse vertex normal
        if line_data[0] == 'vn':
            vn_list.append((float(line_data[1]), float(line_data[2]), float(line_data[3])))

        # parse face
        if line_data[0] == 'f':
            # used for parsing face element data
            def segElementData(ele_str):
                fv = None
                ft = None
                fn = None
                eles = ele_str.strip().split('/')
                if len(eles) == 1:
                    fv = int(eles[0]) - 1
                elif len(eles) == 2:
                    fv = int(eles[0]) - 1
                    ft = int(eles[1]) - 1
                elif len(eles) == 3:
                    fv = int(eles[0]) - 1
                    fn = int(eles[2]) - 1
                    ft = None if eles[1] == '' else int(eles[1]) - 1
                return fv, ft, fn

            fv0, ft0, fn0 = segElementData(line_data[1])
            fv1, ft1, fn1 = segElementData(line_data[2])
            fv2, ft2, fn2 = segElementData(line_data[3])
            f_list.append((fv0, fv1, fv2))
            if ft0 is not None and ft1 is not None and ft2 is not None:
                ft_list.append((ft0, ft1, ft2))
            if fn0 is not None and fn1 is not None and fn2 is not None:
                fn_list.append((fn0, fn1, fn2))

    v_list = np.asarray(v_list)
    vn_list = np.asarray(vn_list)
    vt_list = np.asarray(vt_list)
    vc_list = np.asarray(vc_list)
    f_list = np.asarray(f_list)
    ft_list = np.asarray(ft_list)
    fn_list = np.asarray(fn_list)

    model = {'v': v_list, 'vt': vt_list, 'vc': vc_list, 'vn': vn_list,
             'f': f_list, 'ft': ft_list, 'fn': fn_list}
    return model


# load mesh data
mesh = load_obj_data('./data_sample/results_xxx_xxxxxxxx_xxx_1_F/xxxxx/mesh.obj')
mesh_v = mesh['v']
mesh_v = np.random.permutation(mesh_v)[:10000]  # downsample

# load color/depth
clr = cv.imread('./data_sample/results_xxx_xxxxxxxx_xxx_1_F/xxxxx/color.jpg')
dpt = cv.imread('./data_sample/results_xxx_xxxxxxxx_xxx_1_F/xxxxx/depth.png', cv.IMREAD_UNCHANGED).astype(np.float32) / 1000.0    # mm -> m
print(dpt[dpt.shape[0]//2, dpt.shape[1]//2])
print(np.max(dpt))
dpt = np.clip((dpt-1.0) / 2.5 * 255, 0, 255).astype(np.uint8)
dpt = cv.applyColorMap(dpt, cv.COLORMAP_JET) 

# load world-to-depth transformation
w2d = np.loadtxt('./data_sample/results_xxx_xxxxxxxx_xxx_1_F/xxxxx/cam.txt')

# color intrinsic
c_fx = 1063.8987
c_fy = 1063.6822
c_cx = 954.1103
c_cy = 553.2578

# depth intrinsic
d_fx = 365.4020
d_fy = 365.6674
d_cx = 252.4944
d_cy = 207.7411

# depth-to-color transformation
d2c = np.array([
 [ 9.99968867e-01, -6.80652917e-03, -9.22235761e-03, -5.44798585e-02], 
 [ 6.69376504e-03,  9.99922175e-01, -1.15625133e-02, -3.41685168e-04], 
 [ 9.27761963e-03,  1.14376287e-02,  9.99882174e-01, -2.50539462e-03], 
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00], 
    ])


# dpt = np.pad(dpt, ((28, 28), (64, 64), (0, 0)), mode='constant')
print(dpt.shape)
for v in mesh_v:
    # project to depth image
    v_ = np.array([v[0], v[1], v[2], 1.0]).reshape([4, 1])
    v_ = np.matmul(w2d, v_).reshape([4])
    x = v_[0] / v_[2] * d_fx + d_cx
    y = v_[1] / v_[2] * d_fy + d_cy
    x = int(round(np.clip(x, 0, dpt.shape[1]-1)))
    y = int(round(np.clip(y, 0, dpt.shape[0]-1)))
    dpt[y, x] = np.array([255, 255, 255])

    # project to color image
    v_ = np.matmul(d2c, v_.reshape([4, 1])).reshape([4])
    x = v_[0] / v_[2] * c_fx + c_cx
    y = v_[1] / v_[2] * c_fy + c_cy
    x = int(round(np.clip(x, 0, clr.shape[1]-1)))
    y = int(round(np.clip(y, 0, clr.shape[0]-1)))
    clr[y, x] = np.array([255, 255, 255])

cv.imshow('dpt', dpt)
cv.imshow('clr', clr)
cv.waitKey() 