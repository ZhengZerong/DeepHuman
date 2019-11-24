from __future__ import print_function, absolute_import, division
import numpy as np


def load_ply_data(filename):
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

    lineNum = -1

    class Enum(set):
        def __getattr__(self, name):
            if name in self:
                return name
            raise AttributeError

    State = Enum(["Header", "VertexDef", "FaceDef", "Vertices", "Normals", "Faces"])
    state = State.Header

    orderVertices = -1
    orderIndices = -1

    expectedVertices = 0
    expectedFaces = 0

    vertexCoordsFound = 0
    colorCompsFound = 0
    texCoordsFound = 0
    normalsCoordsFound = 0

    currentVertex = 0
    currentFace = 0
    
    for line in lines:
        lineNum += 1
        line = line.rstrip()
        
        if lineNum == 0:
            if line != 'ply':
                print('Wrong format, expecting ply!!')
                return None
        elif lineNum == 1:
            if line != "format ascii 1.0":
                print("wrong format, expecting 'format ascii 1.0'")
                return

        if 'comment' in line:
            continue

        # HEADER 
        if (state == State.Header or state == State.FaceDef) and line.startswith('element vertex'):
            state = State.VertexDef
            orderVertices = max(orderIndices, 0) + 1
            expectedVertices = int(line[15:])
            # print(state)
            # print(line[15:])
            continue;

        if (state == State.Header or state == State.VertexDef) and line.startswith('element face'):
            state = State.FaceDef
            orderIndices = max(orderVertices, 0) + 1
            expectedFaces = int(line[13:])
            # print(state)
            # print(line[13:])
            continue

        # Vertex Def
        if state == State.VertexDef:

            if line.startswith('property float x') or line.startswith('property float y') or line.startswith(
                    'property float z'):
                vertexCoordsFound += 1
                # print('vertexCoordsFound ' + str(vertexCoordsFound))
                continue

            if line.startswith('property float nx') or line.startswith('property float ny') or line.startswith(
                    'property float nz'):
                normalsCoordsFound += 1
                # print('normalsCoordsFound ' + str(normalsCoordsFound))
                continue

            if line.startswith('property float r') or line.startswith('property float g') or line.startswith(
                    'property float b') or line.startswith('property float a'):
                colorCompsFound += 1
                # print('colorCompsFound ' + str(colorCompsFound))
                floatColor = True
                continue

            if line.startswith('property uchar red') or line.startswith('property uchar green') or line.startswith(
                    'property uchar blue') or line.startswith('property uchar alpha'):
                colorCompsFound += 1
                # print('colorCompsFound ' + str(colorCompsFound))
                floatColor = False
                continue

            if line.startswith('property float u') or line.startswith('property float v'):
                texCoordsFound += 1
                # print('texCoordsFound ' + str(texCoordsFound))
                continue

            if line.startswith('property float texture_u') or line.startswith('property float texture_v'):
                texCoordsFound += 1
                # print('texCoordsFound ' + str(texCoordsFound))
                continue

        # if state==State.FaceDef and line.find('property list')!=0 and line!='end_header':
        #     print('wrong face definition')

        if line == 'end_header':
            # Check that all basic elements seams ok and healthy
            if colorCompsFound > 0 and colorCompsFound < 3:
                print('data has color coordiantes but not correct number of components. Found ' + str(
                    colorCompsFound) + ' expecting 3 or 4')
                return

            if normalsCoordsFound != 3:
                print('data has normal coordiantes but not correct number of components. Found ' + str(
                    normalsCoordsFound) + ' expecting 3')
                return

            if expectedVertices == 0:
                print('mesh loaded has no vertices')
                return

            if orderVertices == -1:
                orderVertices = 9999
            if orderIndices == -1:
                orderIndices = 9999;

            if orderVertices < orderIndices:
                state = State.Vertices
            else:
                state = State.Faces

            continue

        if state == State.Vertices:
            values = line.split()

            # Extract vertex
            v = [0.0, 0.0, 0.0]
            v[0] = float(values.pop(0))
            v[1] = float(values.pop(0))
            if vertexCoordsFound > 2:
                v[2] = float(values.pop(0))
            v_list.append(np.array(v))

            # Extract normal
            if normalsCoordsFound > 0:
                n = [0.0, 0.0, 0.0]
                n[0] = float(values.pop(0))
                n[1] = float(values.pop(0))
                n[2] = float(values.pop(0))
                vn_list.append(np.array(n))

            # Extract color
            if colorCompsFound > 0:
                c = [1.0, 1.0, 1.0, 1.0]
                div = 255.0
                if floatColor:
                    div = 1.0

                c[0] = float(values.pop(0)) / div
                c[1] = float(values.pop(0)) / div
                c[2] = float(values.pop(0)) / div
                if colorCompsFound > 3:
                    c[3] = float(values.pop(0)) / div
                vc_list.append(np.array(c))

            # Extract UVs
            if texCoordsFound > 0:
                uv = [0.0, 0.0]
                uv[0] = float(values.pop(0))
                uv[1] = float(values.pop(0))
                vt_list.append(np.array(uv))

            if len(v_list) == expectedVertices:
                if orderVertices < orderIndices:
                    state = State.Faces
                else:
                    state = State.Vertices
                continue
        
        if state == State.Faces:
            values = line.split()
            numV = int(values.pop(0))

            if numV != 3:
                print("face not a triangle")

            f = np.array([0, 0, 0], dtype=np.int32)
            for i in range(numV):
                index = int(values.pop(0))
                f[i] = index
                
            f_list.append(f)            
            if normalsCoordsFound:
                fn_list.append(f)
            if texCoordsFound:
                ft_list.append(f)

            if currentFace == expectedFaces:
                if orderVertices < orderIndices:
                    state = State.Vertices
                else:
                    state = State.Faces
                continue

            currentFace += 1

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


def save_ply_data(model, filename):
    assert 'v' in model and model['v'].size != 0

    with open(filename, 'wb') as fp:
        lines = 'ply\n'\
                'format ascii 1.0\n'\
                'element vertex ' + str(model['v'].shape[0]) + '\n' \
                'property float x\n' \
               'property float y\n' \
               'property float z\n'

        if 'vn' in model and model['vn'] is not None and  model['vn'].size != 0:
            lines += 'property float nx\n'
            lines += 'property float ny\n'
            lines += 'property float nx\n'
        if 'vc' in model and model['vc'] is not None and model['vc'].size != 0:
            lines += 'property uchar red\n'
            lines += 'property uchar green\n'
            lines += 'property uchar blue\n'
        if 'vt' in model and model['vt'] is not None and model['vt'].size != 0:
            lines += 'property float u\n'
            lines += 'property float v\n'

        if 'f' in model and model['f'] is not None and model['f'].size != 0:
            lines += 'element face ' + str(model['f'].shape[0]) + '\nproperty list uchar int vertex_indices\nend_header\n'
        else:
            lines += 'end_header\n'

        for vi, v in enumerate(model['v']):
            line = '%f %f %f' % (v[0], v[1], v[2])
            if 'vn' in model and model['vn'].size != 0:
                n = model['vn'][vi]
                line += ' %f %f %f' % (n[0], n[1], n[2])
            if 'vc' in model and model['vc'].size != 0:
                c = np.uint8(model['vc'][vi]*255)
                line += ' %i %i %i' % (c[0], c[1], c[2])
            if 'vt' in model and model['vt'].size != 0:
                t = model['vt'][vi]
                line += ' %f %f' % (t[0], t[1])
            lines += line+'\n'

        if 'f' in model and model['f'] is not None and model['f'].size != 0:
            for fi, f in enumerate(model['f']):
                line = '3 %d %d %d' % (f[0], f[1], f[2])
                lines += line+'\n'

        fp.write(lines[:-1])    # delete the last eol
        fp.write('\n')


if __name__ == '__main__':
    mesh = {}
    mesh['v'] = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32)
    mesh['vn'] = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
    mesh['vc'] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]], dtype=np.float32)
    mesh['f'] = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int32)

    save_ply_data(mesh, 'test_save_ply.ply')
    mesh = load_ply_data('test_save_ply.ply')
    mesh['v'] += 1.0
    save_ply_data(mesh, 'test_save_ply2.ply')
