import numpy as np
from scipy import sparse as sparse


def read_data(prefix, foldername):
    folderpath = f'{prefix}/{foldername}'
    with open(f'{folderpath}/graph.meta.txt', 'r') as f:
        _, _, vid_size, eid_size = f.read().split()[:4]
    datatypes = {
        '4': np.int32,
        '8': np.int64
    }
    indptr = np.fromfile(
        f'{folderpath}/graph.vertex.bin', dtype=datatypes[eid_size])
    indices = np.fromfile(
        f'{folderpath}/graph.edge.bin', dtype=datatypes[vid_size])
    return indptr, indices
    

def k_core(indptr, indices, k):
    n_vrtx = len(indptr) - 1
    degs = indptr[1:] - indptr[:-1]

    # nbhds = {i: set(indices[indptr[i]:indptr[i+1]]) for i in range(n_vrtx)}
    queue = np.where(degs < k)[0].tolist()
    deleted = set(queue)
    
    while len(queue) > 0:
        vrtx = queue.pop()
        nbrs = indices[indptr[vrtx]:indptr[vrtx+1]]
        degs[nbrs] -= 1
        
        if len(queue) == 0:
            queue = set(np.where(degs < k)[0]) - deleted
            deleted |= queue
    
    return deleted, set(range(n_vrtx)) - deleted


def cc(indptr, indices):
    n_vrtx = len(indptr) - 1
    partitions = []
    remaining = set(range(n_vrtx))

    nbhds = {i: set(indices[indptr[i]:indptr[i+1]]) for i in range(n_vrtx)}

    while len(remaining) > 0:
        start = remaining.pop()
        queue = [start]
        visited = {start}
        while len(queue) > 0:
            v = queue.pop()
            queue.extend(nbhds[v] - visited)
            visited |= nbhds[v]
        remaining -= visited
        partitions.append(visited)

    return partitions


if __name__ == '__main__':
    N, F = read_data('./inputs', 'citeseer')