from sys import argv
from time import time

from collections import defaultdict
from copy import deepcopy
from itertools import permutations
import numpy as np
from scipy import sparse

from common import read_data


def pattern_sym_ord(indptr, indices):
    '''
    Given the edge list of a pattern, generate a symmetry ordering.

    Parameters
    ----------
    `indptr`, `indices` : np array
        the neighbors of `i` are stored in `indices[indptr[i]:indptr[i+1]]`.
        assumes symmetric.
    '''
    n_vertices = len(indptr) - 1
    # WL
    deg = indptr[1:] - indptr[:-1]
    d_prev = {i: deg[i] for i in range(n_vertices)}
    d_curr = {}
    nbhd_map = {i: set(indices[indptr[i]:indptr[i+1]])
                for i in range(n_vertices)}
    
    for i in range(n_vertices):
        for j in range(n_vertices):
            d_curr[j] = (d_prev[j], tuple(
                sorted([hash(d_prev[nbr]) for nbr in nbhd_map[j]])))
        d_prev = deepcopy(d_curr)
    
    # find potential symmetries
    grouping = 0
    selected = set()
    symmetry = np.empty((n_vertices), dtype=int)
    for i in range(n_vertices):
        if i not in selected:
            symmetry[i] = grouping
            for j in range(i+1, n_vertices):
                if j not in selected:
                    if d_curr[i] == d_curr[j]:
                        symmetry[j] = grouping
                        selected.add(j)
            grouping += 1

    # TODO: do i need to validate symmetries?
    print('symmetry:', symmetry)

    # vrtx ordering
    adj = sparse.csr_matrix((np.ones(indices.shape), indices, indptr),
                            shape=(n_vertices, n_vertices),
                            dtype=int).toarray()

    v_list = list(range(n_vertices))
    opt_val = np.zeros(n_vertices, dtype=int)
    opt_perm = None

    first_ind_filt = set()
    for sym in np.unique(symmetry):
        first_ind_filt.add(np.where(symmetry==sym)[0][0])
    
    for perm in permutations(v_list):
        if perm[0] in first_ind_filt:
            _perm_adj = _permute(adj, perm)
            val = np.sum(np.triu(_perm_adj), axis=0)
            diff = val - opt_val
            diff_ind = np.nonzero(diff)[0]
            if len(diff_ind) > 0:
                if diff[diff_ind[0]] > 0:
                    opt_val = val
                    opt_perm = perm

    vrtx_ord = list(opt_perm)
    print('vrtx order:', vrtx_ord)
    
    # symm ordering
    path_vids = set()
    symm_ord = defaultdict(list)
    symmetry = symmetry[vrtx_ord]
    for i in range(n_vertices):
        vid = vrtx_ord[i]
        edges = path_vids & nbhd_map[vid]
        def is_equiv(_v):
            return _v > i and (path_vids & nbhd_map[vrtx_ord[_v]]) == edges
        symm_vids = np.where(symmetry == symmetry[i])[0]
        symm_vids = filter(is_equiv, symm_vids)
        for _v in symm_vids:
                symm_ord[_v].append(i)
        
        path_vids.add(vid)      # vrtx_ord[:i]

    adj_map = {
        i: frozenset(np.where(adj[i][:i] != 0)[0])
        for i in range(len(adj))
    }

    return vrtx_ord, symm_ord, adj_map


def _permute(arr, perm):
    return arr[perm,:][:,perm]


def generate_ccode(pattern_adj, pattern_symm):
    n_vertices = len(pattern_adj)

    s = []
    s.append('  vidType lower = 0;')
    s.append('  for (vidType v0 = 0; v0 < g.V(); v0 ++) {')
    indent = ' '*4

    buff = {}
    for vrtx in range(1, n_vertices):
        var_name = _add_var_name(elts={vrtx-1}, buff=buff)
        s.append('{}VertexSet {} = g.N(v{});'.format(indent, var_name, vrtx-1))
        
        nbrs = pattern_adj[vrtx]
        nonnbrs = set(range(vrtx)) - nbrs
        has_nonnbrs = len(nonnbrs) > 0

        keys_to_merge = [_set_factor(buff, nbrs, nbrs)]
        if has_nonnbrs:
            keys_to_merge.append(_set_factor(buff, nonnbrs, nonnbrs))

        # merge stored vertexsets
        vset_names = []
        for keys in keys_to_merge:
            elts = keys[0]
            vset_name = buff[keys[0]]
            for i in range(1, len(keys)):
                elts |= keys[i]
                var_name = _add_var_name(elts=elts, buff=buff)
                s.append('{}VertexSet {} = intersection_set({}, {});'.format(
                    indent, var_name, vset_name, buff[keys[i]]))
                vset_name = var_name
            vset_names.append(vset_name)
        
        # get lower bound of vid
        symm = pattern_symm.get(vrtx, [])
        lower = len(symm) > 0
        if lower:
            if len(symm) == 1:
                s.append('{}lower = v{};'.format(indent, symm[0]))
            else:
                s.append('{}lower = min(v{}, v{});'.format(indent, symm[0], symm[1]))
                for _v in symm[2:]:
                    s.append('{}lower = min(lower, v{});'.format(indent, _v))
            LOWER_BOUND_STR = ', lower'
        else:
            LOWER_BOUND_STR = ''

        # increment counter
        if vrtx == n_vertices-1:
            if has_nonnbrs:
                s.append('{}counter += (uint64_t)difference_num({}, {}{});'.format(
                    indent, vset_names[0], vset_names[1], LOWER_BOUND_STR))
            elif lower:
                    s.append('{}counter += (uint64_t)bounded({}, lower).size();'.format(
                        indent, vset_names[0]))
            else:
                s.append('{}counter += (uint64_t){}.size();'.format(
                    indent, vset_names[0]))
        # loop
        else:
            if has_nonnbrs:
                s.append('{}for (auto v{} : difference_set({}, {}{})) {{'.format(
                    indent, vrtx, vset_names[0], vset_names[1], LOWER_BOUND_STR
                ))
            elif lower:
                s.append('{}for (auto v{} : bounded({}, lower)) {{'.format(
                    indent, vrtx, vset_names[0]
                ))
            else:
                s.append('{}for (auto v{} : {}) {{'.format(
                    indent, vrtx, vset_names[0]
                ))
        indent += ' '*2

    indent = indent[2:]
    while (len(indent) > 2):
        indent = indent[2:]
        s.append('{}}}'.format(indent))

    print()
    print('\n'.join(s))
    return '\n'.join(s)


def _set_factor(buff, elts, remaining):
    remaining = frozenset(remaining)
    if len(remaining) == 0:
        return []
    if remaining in buff:
        return [remaining]
    max_size = 0
    max_subset = None
    for subset in buff:
        if subset.issubset(elts):
            size = len(subset & remaining)
            if size > max_size:
                max_subset = subset
                max_size = size
    factors = _set_factor(buff, elts, remaining - max_subset)
    factors.append(max_subset)
    return factors

def _add_var_name(elts, buff):
    var_name = ''.join(['y{}'.format(elt) for elt in sorted(list(elts))])
    buff[frozenset(elts)] = var_name
    return var_name

def count_pattern(indptr, indices, pattern_adj, pattern_symm):
    start_time = time()

    n_vertices = len(indptr) - 1

    # convert edge list to a map from a node to the set of its neighbors,
    # to avoid repeated set conversions
    nbhd_map = {i: set(indices[indptr[i]:indptr[i+1]])
                for i in range(n_vertices)}
    # count cliques using the tree / for loop method
    count = _pattern_recur(nth=0,
                           n_graph_vrtx=n_vertices,
                           n_pattern_vrtx=len(pattern_adj),
                           nbhd_map=nbhd_map,
                           pattern_adj=pattern_adj,
                           pattern_symm=pattern_symm,
                           hist=[])

    return count
    

def _pattern_recur(nth, n_graph_vrtx, nbhd_map,
                   n_pattern_vrtx, pattern_adj, pattern_symm, hist):
    adj = pattern_adj[nth]
    nbhd = set(range(n_graph_vrtx)).intersection(
        *[nbhd_map[hist[i]] for i in adj])
    nbhd = nbhd - set().union(
        *[nbhd_map[hist[i]] for i in set(range(nth)) - adj])

    def filt(vrtx):
        return vrtx > max([hist[x] for x in pattern_symm[nth]], default=-1)
    
    if nth == n_pattern_vrtx-1:    
        return len(list(filter(filt, nbhd)))
    
    count = 0
    for vrtx in list(filter(filt, nbhd)):
            hist.append(vrtx)
            count += _pattern_recur(
                nth+1, n_graph_vrtx, nbhd_map,
                n_pattern_vrtx, pattern_adj, pattern_symm, hist)
            hist.pop(-1)
    return count


if __name__ == '__main__':
    if len(argv) < 2:
        print('format: [./pattern.py pattern]')
    else:
        start_time = time()
        pattern = argv[1]
        # dataset = argv[2]
        N_patt, F_patt = read_data('./codegen/input_patterns', pattern)
        # N_data, F_data = read_data('./GraphMiner/inputs', dataset)
        print('read data: {:.6f}s'.format(time() - start_time))
        
        start_time = time()
        vrtx_ord, symm_ord, adj_map = pattern_sym_ord(N_patt, F_patt)
        print('find order: {:.6f}s'.format(time() - start_time))
        print('symmetry order:', symm_ord)
        print('adjacency map:', adj_map)

        generate_ccode(adj_map, symm_ord)