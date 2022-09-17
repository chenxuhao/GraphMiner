import os
import subprocess

import numpy as np


def bin_pattern(indices, indptr, name):
    folder_path = './codegen/input_patterns/{}'.format(name)
    subprocess.run(['mkdir', folder_path], check=True)
    if indices.dtype == np.int64:
        vrtx_size = 8
    elif indices.dtype == np.int32:
        vrtx_size = 4
    else:
        print('unrecognized dtype:', indices.dtype)
    if indptr.dtype == np.int64:
        edge_size = 8
    elif indptr.dtype == np.int32:
        edge_size = 4
    else:
        print('unrecognized dtype:', indptr.dtype)
    
    meta_filepath = os.path.join(folder_path, 'graph.meta.txt')
    subprocess.run(['touch', meta_filepath], check=True)
    with open(meta_filepath, 'w') as f:
        f.write('\n'.join([
            str(len(indptr)),
            str(len(indices)),
            '\t'.join([str(vrtx_size), str(edge_size), '1', '2']),
            str(max(indptr[1:] - indptr[:-1])),
            '0', '0', '0'
        ]))
    
    vrtx_filepath = os.path.join(folder_path, 'graph.vertex.bin')
    subprocess.run(['touch', vrtx_filepath], check=True)
    indptr.tofile(vrtx_filepath)

    edge_filepath = os.path.join(folder_path, 'graph.edge.bin')
    subprocess.run(['touch', edge_filepath], check=True)
    indices.tofile(edge_filepath)


if __name__ == '__main__':
    bin_pattern(
        indices=np.array([1,2,0,2,3,0,1,3,1,2]),
        indptr=np.cumsum(np.array([0,2,3,3,2])),
        name='diamond'
    )
    