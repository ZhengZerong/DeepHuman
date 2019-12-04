from smpl_webuser.serialization import load_model
import numpy as np


## Read smpl parameter
with open('./data_sample/smpl_params.txt', 'r') as fp:
    lines = fp.readlines()
    lines = [l[:-2] for l in lines]     # remove '\r\n'
    
    betas_data = filter(lambda s: len(s)!=0, lines[1].split(' '))
    betas = np.array([float(b) for b in betas_data])
    
    root_mat_data = lines[3].split(' ') + lines[4].split(' ') +\
                    lines[5].split(' ') + lines[6].split(' ')
    root_mat_data = filter(lambda s: len(s)!=0, root_mat_data)
    root_mat = np.reshape(np.array([float(m) for m in root_mat_data]), (4, 4))
    root_rot = root_mat[:3, :3]
    root_trans = root_mat[:3, 3]

    theta_data = lines[8:80]
    theta = np.array([float(t) for t in theta_data])

## Load SMPL model (here we load the male model)
## Make sure path is correct
m = load_model( './smpl_webuser/basicModel_f_lbs_10_207_0_v1.0.0.pkl' )

## Apply shape & pose parameters
m.pose[:] = theta
m.betas[:] = betas

## Apply root transformation
verts = m.r
verts = np.matmul(verts, root_rot.transpose()) + np.reshape(root_trans, (1, -1))

## Write to an .obj file
outmesh_path = './hello_smpl.obj'
with open( outmesh_path, 'w') as fp:
    for v in verts:
        fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

    for f in m.f+1: # Faces are 1-based, not 0-based in obj files
        fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

## Print message
print '..Output mesh saved to: ', outmesh_path 