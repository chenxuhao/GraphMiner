from enum import Enum
from sys import argv
from time import time
from copy import deepcopy
from collections import Counter, defaultdict
from itertools import product, permutations

import numpy as np

from common import read_data


SPARSITY = 0.001


COST_SCALE = {
    'storage': 3,
    'intersect': 1,
    'sort': 1,
    'valid': 1
}


def get_cost(**kwargs):
    return sum([COST_SCALE.get(k, 1) * v for (k, v) in kwargs.items()])


class status(Enum):
    CONFLICT = 0
    SUCCESS = 1
    REDUNDANT = 2


class hybrid_action(Enum):
    DEFAULT = 0         # edge / other existing things
    EXTEND = 1          # extend, original, connectivity, constraints
    MERGE = 2           # merge, sub_0, sub_1, [lopairings], constraints


def WL(n_vrtx, indptr, indices):
    # not conclusive? what's k-WL
    deg = indptr[1:] - indptr[:-1]
    z_prev = {i: deg[i] for i in range(n_vrtx)}     # some multiset embedding
    z_curr = {}
    nbhd_map = {i: set(indices[indptr[i]:indptr[i+1]])
                for i in range(n_vrtx)}

    for i in range(n_vrtx):
        for j in range(n_vrtx):
            z_curr[j] = (z_prev[j], tuple(
                sorted([hash(z_prev[nbr]) for nbr in nbhd_map[j]])))
        z_prev = deepcopy(z_curr)
    return z_curr


def inverse_permutation(n_vrtx, perm):
    trg = np.empty(n_vrtx, dtype=int)
    trg[perm] = np.arange(n_vrtx)


class Pattern():
    def __init__(self, adj=None, forw_indptr=None, forw_indices=None):
        if adj is not None:
            self.n_vrtx = len(adj)
            self.n_edges = sum(adj)
            self.adj = adj
            self.forw_indptr = np.cumsum([0, *np.sum(adj, axis=1)])
            self.forw_indices = np.nonzero(adj)[1]
        else:
            self.n_vrtx = len(forw_indptr) - 1
            self.n_edges = len(forw_indices)
            self.forw_indptr = forw_indptr
            self.forw_indices = forw_indices
            self.adj = np.zeros((self.n_vrtx, self.n_vrtx), dtype=int)
            for i in range(self.n_vrtx):
                self.adj[i, forw_indices[forw_indptr[i]:forw_indptr[i+1]]] = 1
        self.back_indptr = np.cumsum([0, *np.sum(self.adj, axis=0)])
        self.back_indices = np.nonzero(self.adj.T)[1]

        self.id = get_matrix_hashable(self.adj)
        self.id_adj_perm = np.arange(self.n_vrtx)
        self.automorphisms = None
        self.subgraphs = defaultdict(set)
        self.reverse_map = {}       # maps vsets to subgraph_ids

    def __repr__(self):
        adj_str = str(self.adj).replace('\n','')
        return f'<Pattern> {adj_str}'

    def set_id(self, new_id, get_permutation=False):
        if new_id == self.id:
            return
        self.id = new_id
        if get_permutation:
            trg_adj = np.array(new_id[1], dtype=int).reshape(new_id[0])
            for perm in permutations(range(self.n_vrtx)):
                if self.adj[perm, :][:, perm] == trg_adj:
                    self.id_adj_perm = perm
                    return
            self.id_adj_perm = None
            return

    def get_symmetry(self):
        wl_label = WL(self.n_vrtx, self.forw_indptr, self.forw_indices)
        grouping = 0
        selected = set()
        symmetry = np.empty((self.n_vrtx), dtype=int)
        for i in range(self.n_vrtx):
            if i not in selected:
                symmetry[i] = grouping
                for j in range(i+1, self.n_vrtx):
                    if j not in selected:
                        if wl_label[i] == wl_label[j]:
                            symmetry[j] = grouping
                            selected.add(j)
                grouping += 1
        return symmetry

    def get_automorphisms(self):
        if self.automorphisms is not None:
            return self.automorphisms
        symmetry = self.get_symmetry()

        n_cycles = max(symmetry)
        symm_indx = []
        for i in range(n_cycles+1):
            symm_indx.append(np.where(symmetry == i)[0])
        symm_perms = [permutations(s) for s in symm_indx]

        automorphisms = []
        for cycle_prod in product(*symm_perms):
            permutation = np.zeros(self.n_vrtx, dtype=int)
            for i in range(n_cycles+1):
                permutation[symm_indx[i]] = cycle_prod[i]
            if np.all(self.adj[permutation, :][:, permutation] == self.adj):
                automorphisms.append(permutation)

        self.automorphisms = np.array(automorphisms)
        return self.automorphisms

    def get_nonauto_combs(self, k):
        # generate all combinations (ordered) of size k that don't have
        # automorphic mappings (pairwise)
        automorphisms = self.get_automorphisms()
        unfilt_combs = set(permutations(range(self.n_vrtx), k))
        unique_combs = []
        while len(unfilt_combs) > 0:
            comb = unfilt_combs.pop()
            unfilt_combs -= set(tuple(x) for x in automorphisms[:, comb])
            unique_combs.append(comb)

        return unique_combs

    def add_subgraph(self, subgraph, instances, check_duplicate=False):
        # generally only use if adding from scratch, so mostly just edges
        sub_id = subgraph.id
        assert len(instances) > 0, \
            'expected nonempty list of instances or None'

        if check_duplicate:
            try:
                sub_id = self.reverse_map[frozenset(instances[0])]
                return sub_id
            except KeyError:
                pass

        instances = set(instances)
        self.subgraphs[sub_id] |= instances
        for x in instances:
            self.reverse_map[frozenset(x)] = sub_id

        return sub_id

    def get_merged_instances(self, sub_0, paired_0, sub_1, paired_1,
                             new_sub, reindex=None, sym_pairs=None):
        '''
        Each id corresponds to one adjacency matrix. each subgraph instance is
        stored only once - there are no symmetric permutations. cost related
        things, like set sizes and symmetries, are stored elsewhere.

        Parameters
        ----------
        `pairing` : tuple of 1d array-like
            Where `vset0[pairing[0][i]] == vset1[pairing[1][i]]`, i.e. where
            the vertex sets should intersect.
        `symm_ind` : numpy int array
            A 2D-array of the form
                `[[subgraph1_indexing], [subgraph2_indexing]]`.
            For example,
                `symm_ind = [[2, 3], [0, 0]]
            means that vertices 2 and 3 of the first subgraph are both larger
            (have higher vid) than vertex 0 of the second subgraph.

        Returns
        ----------
        `d`: Counter { vertex_set : int }
            The keys should be enough, the count is returned for multiplicity
            checking purposes. Given the symmetry order, we expect to count
            each merged instance once (in any query!)
        '''

        new_subgraphs = Counter()

        if reindex is None:
            reindex = _get_merged_reindex(sub_0, sub_1, paired_0, paired_1)

        first_set = True
        # for each pair of sub0_id and sub1_id instances
        for vset0 in self.subgraphs[sub_0.id]:
            vset0_arr = np.array(list(vset0))
            new_vset = np.empty(new_sub.n_vrtx, dtype=int)
            new_vset[range(sub_0.n_vrtx)] = vset0_arr
            for vset1 in self.subgraphs[sub_1.id]:
                vset1_arr = np.array(list(vset1))
                # check if the vid intersection is valid
                if not np.all(vset0_arr[list(paired_0)] == vset1_arr[list(paired_1)]):
                    continue
                new_vset[reindex] = vset1_arr
                # check if vrtx-induced connectivity is valid
                if not np.all(self.adj[new_vset, :][:, new_vset] == new_sub.adj):
                    continue
                # check symmetry order
                if sym_pairs is not None:
                    if not np.all(new_vset[sym_pairs[0]] < new_vset[sym_pairs[1]]):
                        continue

                new_subgraphs[tuple(new_vset)] += 1

                # check if this structure has already been examined
                if first_set:
                    try:
                        new_id = self.reverse_map[frozenset(new_vset)]
                        # TODO: might need to return reindexing for both subs
                        return new_id, self.subgraphs[new_id]
                    except KeyError:
                        first_set = False

        if first_set:
            return -1, set()

        # update stored subgraphs
        self.subgraphs[new_sub.id] |= set(new_subgraphs.keys())
        # returns new vsets and counts
        return new_sub.id, new_subgraphs


    def get_extended_instances(self, old_sub, new_sub, connectivity, sym_pairs=None):
        new_subgraphs = Counter()
        first_set = True

        # for each pair of sub0_id and sub1_id instances
        for vset in self.subgraphs[old_sub.id]:
            vset_list = list(vset)
            new_vset = np.empty(new_sub.n_vrtx, dtype=int)
            new_vset[range(old_sub.n_vrtx)] = vset_list
            for vrtx in range(self.n_vrtx):
                if (vrtx not in vset and \
                        np.all(self.adj[vrtx, vset_list] == connectivity)):
                    new_vset[-1] = vrtx
                    new_subgraphs[tuple(new_vset)] += 1

                    # check if this structure has already been examined
                    if first_set:
                        try:
                            new_id = self.reverse_map[frozenset(new_vset)]
                            return new_id, self.subgraphs[new_id]
                        except KeyError:
                            first_set = False

        # no instances found
        if first_set:
            return -1, set()

        # update stored subgraphs
        self.subgraphs[new_sub.id] |= set(new_subgraphs.keys())
        # returns new vsets and counts
        return new_sub.id, new_subgraphs


    def extend_vrtx(self, sub_id, sub_size,
                    new_id, new_adj, new_sym):

        new_subgraphs = Counter()
        sym_pairs = np.nonzero(new_sym)

        first_set = True

        # for each pair of sub0_id and sub1_id instances
        for old_vset in self.subgraphs[sub_id]:
            old_vset_list = list(old_vset)
            new_vset = np.empty(sub_size + 1, dtype=int)
            new_vset[range(sub_size)] = old_vset_list

            for vrtx in set(range(self.n_vrtx)) - old_vset:
                # check if vrtx-induced connectivity is valid
                if not np.all(self.adj[-1, old_vset_list] == new_adj[-1, :-1]):
                    continue
                new_vset[-1] = vrtx
                # check symmetry order
                if not np.all(new_vset[sym_pairs[0]] < new_vset[sym_pairs[1]]):
                    continue

                new_subgraphs[tuple(new_vset)] += 1

                if first_set:
                    # check if this structure has already been examined
                    try:
                        old_id = self.reverse_map[frozenset(new_vset)]
                        return old_id, self.subgraphs[old_id]
                    except KeyError:
                        first_set = False

        # update stored subgraphs
        self.subgraphs[new_id] |= set(new_subgraphs.keys())

        # returns new vsets and counts
        return new_id, new_subgraphs

    def possible_extensions(self, sub):
        # returns a set of connectivities
        ret = set()
        # list possible connectivities for appending 1 vertex
        for vset in self.subgraphs[sub.id]:
            for vrtx in set(range(self.n_vrtx)) - set(vset):
                conn = self.adj[vrtx, vset]
                if sum(conn) != 0:
                    ret.add(tuple(self.adj[vrtx, vset]))
        return ret


def get_matrix_hashable(mat):
    return (mat.shape, tuple(mat.flatten()))


def _get_merged_reindex(sub_0, sub_1, paired_0, paired_1):
    adj1_reindex = [None] * sub_1.n_vrtx
    for pair in zip(paired_0, paired_1):
        adj1_reindex[pair[1]] = pair[0]
    counter = sub_0.n_vrtx
    for i in range(sub_1.n_vrtx):
        if adj1_reindex[i] is None:
            adj1_reindex[i] = counter
            counter += 1
    return adj1_reindex

class Partial_Ordering():
    def __init__(self):
        self.root = None
        # self.child_map = defaultdict(set)
        self.ancst_map = defaultdict(set)
        self.desct_map = defaultdict(set)

    def add_relation(self, parent, child):
        if child in self.desct_map[parent]:
            return status.REDUNDANT
        if parent in self.desct_map[child]:
            return status.CONFLICT

        # self.child_map[parent].add(child)
        self.desct_map[parent].add(child)
        self.ancst_map[child].add(parent)

        self.ancst_map[child] |= self.ancst_map[parent]
        self.desct_map[parent] |= self.desct_map[child]

        return status.SUCCESS

    def subtree(self, reindex):
        # nodes is a list of node ids
        # returns a list of [reindexed] symmetry pairs involving the given nodes
        reindex_map = {reindex[i]: i for i in range(len(reindex))}
        def reindex_many(y): return [reindex_map[x] for x in y if x in reindex]

        ret = []
        for vrtx in self.desct_map:
            if vrtx in reindex:
                vrtx_r = reindex_map[vrtx]
                children = reindex_many(self.desct_map[vrtx])
                ret.extend([(vrtx_r, child) for child in children])

        return ret


def get_merged_pattern(sub_0, paired_0, sym_0, sub_1, paired_1, sym_1, reindex=None):
    # check adj compatibility
    assert np.all(sub_0.adj[paired_0, :][:, paired_0] == sub_1.adj[paired_1, :][:, paired_1]), \
        'the induced subgraphs of shared vertices are different.'

    # TODO: this part feels redundant! check same induced sym
    sym_mat_0 = np.zeros((sub_0.n_vrtx, sub_0.n_vrtx), dtype=bool)
    sym_mat_1 = np.zeros((sub_1.n_vrtx, sub_1.n_vrtx), dtype=bool)
    sym_0 = np.array(sym_0)
    sym_1 = np.array(sym_1)
    sym_mat_0[sym_0[:, 0], sym_0[:, 1]] = True
    sym_mat_1[sym_1[:, 0], sym_1[:, 1]] = True

    assert np.all(sym_mat_0[paired_0, :][:, paired_0] == sym_mat_1[paired_1, :][:, paired_1]), \
        'the induced symmetries of shared vertices are different.'

    if reindex is None:
        reindex = _get_merged_reindex(sub_0, sub_1, paired_0, paired_1)
    # create and fill new adj matrix
    new_size = sub_0.n_vrtx + sub_1.n_vrtx - len(paired_0)
    new_adj = np.zeros((new_size, new_size), dtype=int)

    new_adj[:sub_0.n_vrtx, :sub_0.n_vrtx] = sub_0.adj
    new_adj[np.repeat(reindex, len(reindex)), list(reindex) * len(reindex)] = sub_1.adj.flatten()
    return Pattern(new_adj)


def get_extended_pattern(old_sub, connectivity):
    new_size = old_sub.n_vrtx + 1
    new_adj = np.zeros((new_size, new_size), dtype=int)

    new_adj[:old_sub.n_vrtx, :old_sub.n_vrtx] = old_sub.adj
    new_adj[-1, :-1] = connectivity
    new_adj[:-1, -1] = connectivity

    return Pattern(new_adj)


def get_restricted_perms(sym_pairs, perms):
    sym_poset = Partial_Ordering()
    for pair in sym_pairs:
        perms = _update_perm_mask(perms, sym_poset, pair[0], pair[1])

    return perms


def update_symmetry(sym_pairs, perms):
    sym_poset = Partial_Ordering()
    del_pairs = []
    add_pairs = []

    n_vrtx = len(perms[0])

    for pair in sym_pairs:
        # check validity in perms
        # (exist mapping from pair[0] -> pair[1])
        if not np.any(perms[:, pair[0]] == pair[1]):
            del_pairs.append(pair)
            continue
        # iterate through perms, check whether perm[pair[0]] > perm[pair[1]]
        # can be added to the poset. if not, delete perm.
        perms = _update_perm_mask(perms, sym_poset, pair[0], pair[1])

    while len(perms) > 1:
        diff_ind = np.nonzero(perms[0] - perms[1])[0][0]
        # need positions of the first differing vids, in the first remaining permutation
        inv_p = np.empty(n_vrtx, dtype=int)
        inv_p[perms[0]] = np.arange(n_vrtx)

        pair_0_val = perms[0][diff_ind]
        pair_1_val = perms[1][diff_ind]
        pair = (inv_p[pair_0_val], inv_p[pair_1_val])
        add_pairs.append(pair)

        perms = _update_perm_mask(perms, sym_poset, pair[0], pair[1])

    return del_pairs, add_pairs, sym_poset


def _update_perm_mask(perms, sym_poset, pair_0, pair_1):
    perm_mask = np.ones(len(perms), dtype=bool)
    for i, perm in enumerate(perms):
        if perm_mask[i]:
            res = sym_poset.add_relation(
                parent=perm[pair_0],
                child=perm[pair_1]
            )
            if res == status.CONFLICT:
                perm_mask[i] = False
                continue
    perms = perms[perm_mask]
    return perms


def _mapped(nested_list, the_map):
    if isinstance(nested_list, (list, tuple)):
        return [_mapped(elt, the_map) for elt in nested_list]
    return the_map[nested_list]


def get_merge_symmetry(sub_0, paired_0, sym_0,
                          sub_1, paired_1, sym_1,
                          new_sub, reindex=None):
    if reindex is None:
        reindex = _get_merged_reindex(
            sub_0, sub_1, paired_0, paired_1)

    # # filter existing symmetries
    reindex_map = {i: reindex[i] for i in range(len(reindex))}
    # list of sym pairs for sub1
    reindex_sym_1 = _mapped(sym_1, reindex_map)
    sym_pairs = sym_0 + reindex_sym_1
    perms = new_sub.get_automorphisms()

    del_sym_pairs, add_sym_pairs, sym_poset = update_symmetry(
        sym_pairs=sym_pairs, perms=perms
    )

    sym_pairs = list(filter(lambda x: x not in del_sym_pairs, sym_pairs))
    sym_pairs.extend(add_sym_pairs)

    _logger(
        block_title='[outer merge sym]',
        adj_0=sub_0.adj,
        adj_1=sub_1.adj,
        new_adj=new_sub.adj,
        del_sym=del_sym_pairs,
        add_sym=add_sym_pairs
    )

    return sym_pairs, add_sym_pairs, del_sym_pairs, sym_poset


def get_extend_symmetry(old_sub, old_sym, new_sub):
    perms = new_sub.get_automorphisms()

    del_sym_pairs, add_sym_pairs, sym_poset = update_symmetry(
        sym_pairs=old_sym, perms=perms
    )

    sym_pairs = list(filter(lambda x: x not in del_sym_pairs, old_sym))
    sym_pairs.extend(add_sym_pairs)

    # _logger(
    #     block_title='[outer extend sym]',
    #     old_adj=old_sub.adj,
    #     new_adj=new_sub.adj,
    #     del_sym=del_sym_pairs,
    #     add_sym=add_sym_pairs
    # )

    return sym_pairs, add_sym_pairs, del_sym_pairs, sym_poset


class Cost():
    def __init__(self, pattern, **kwargs):
        '''
        kwargs are global variables (inferred from target graph).
        '''
        self.pattern = pattern
        self.subgraph_info = {}

        self.V = kwargs.get('V', 10000)
        self.E = kwargs.get('E', self.V * 10)

        self.density = kwargs.get('density', self.E / (self.V * self.V))
        self.max_deg = kwargs.get('max_deg', int(np.sqrt(self.V)))
        self.avg_deg = kwargs.get('avg_deg', self.E / self.V)

    def add_subgraph(self, new_sub, info):
        self.subgraph_info[new_sub.id] = info

    def get_extend_costs(self, old_sub, new_sub, n_paired):
        # n_paired is the sum of connectivity

        old_info = self.subgraph_info[old_sub.id]
        old_count = old_info['count']

        if new_sub.id in self.subgraph_info:
            new_info = self.subgraph_info[new_sub.id]
        else:
            new_info = {}
            old_multi = old_info['multiplicity']
            new_multi = len(new_sub.get_automorphisms())

            old_count = old_info['count']
            new_count = old_count * old_multi * self.V / \
                (self.density ** n_paired) / new_multi

            # avoid dividing by 0 or smth
            new_info['count'] = max(10, new_count)
            new_info['multiplicity'] = new_multi

            self.subgraph_info[new_sub.id] = new_info

        new_count = new_info['count']
        costs = {
            # compare shared entries
            'join_intersect': n_paired * self.avg_deg * old_count,
            # only unit sorting
            'sort': new_count * np.log(new_count),
            # only unit storage - num vertices stored unknown
            'storage': new_count
        }

        return costs

    def get_merge_costs(self, sub_0, sub_1, new_sub):

        n_paired = sub_0.n_vrtx + sub_1.n_vrtx - new_sub.n_vrtx

        info_0 = self.subgraph_info[sub_0]
        info_1 = self.subgraph_info[sub_1]

        count_0 = info_0['count']
        count_1 = info_1['count']

        if new_sub.id in self.subgraph_info:
            new_info = self.subgraph_info[new_sub.id]
        else:
            new_info = {}
            multi_0 = info_0['multiplicity']
            multi_1 = info_1['multiplicity']

            new_multi = len(new_sub.get_automorphisms())

            # the following cancels out:
            # prob0 /= self.V ** size0, prob1 /= self.V ** size1
            # new_count = prob0 * prob1 / (self.V ** new_size) / multi
            prob0 = count_0 * multi_0
            prob1 = count_1 * multi_1
            new_count = prob0 * prob1 / (self.V ** n_paired) / new_multi

            # avoid dividing by 0 or smth
            new_info['count'] = max(10, new_count)
            new_info['multiplicity'] = new_multi

            self.subgraph_info[new_sub.id] = new_info

        new_count = new_info['count']
        costs = {
            # compare shared entries
            'join_comp': count_0 + count_1,
            # validate remaining entries
            'join_valid': new_count * (sub_0.n_vrtx-n_paired-1) * (sub_1.n_vrtx-n_paired-1),
            # only unit sorting
            'sort': new_count * np.log(new_count),
            # only unit storage - num vertices stored unknown
            'storage': new_count
        }

        return costs

def plan_dp(pattern, **kwargs):

    dp_tracker = {k: {} for k in range(2, pattern.n_vrtx+1)}
    pattern_cost = Cost(pattern, **kwargs)

    # base case: edge
    # make edge entry
    edge_adj = np.array([[0, 1], [1, 0]])
    edge_sym = [(0, 1)]
    edge_obj = Pattern(edge_adj)

    edge_set = set(zip(*np.nonzero(pattern.adj)))
    pattern.add_subgraph(edge_obj, instances=edge_set)

    # structure related information
    pattern_cost.add_subgraph(edge_obj, info={
        'count': pattern_cost.E,
        'multiplicity': 2,
        'storage': pattern_cost.E * 2
    })

    dp_tracker[2][edge_obj.id] = {
        'obj': edge_obj,
        'sym': edge_sym,
        'calc_cost': COST_SCALE['storage'] * pattern_cost.E,
        'step': {
            'type': hybrid_action.DEFAULT,
            'name': 'edge'
        },
        'cache': {}         # TODO: also storage cache
    }

    for k in range(3, pattern.n_vrtx+1):
        for dp_info in dp_tracker[k-1].values():
            old_sub = dp_info['obj']
            exts = pattern.possible_extensions(old_sub)
            for ext in exts:
                # each ext is an array for the connectivity of new node to current subgraph
                new_sub = get_extended_pattern(
                    old_sub=old_sub, connectivity=ext)

                new_sym, add_sym, del_sym, sym_poset = \
                    get_extend_symmetry(
                        old_sub=old_sub,
                        old_sym=dp_info['sym'],
                        new_sub=new_sub
                    )

                structure_id, instances = \
                    pattern.get_extended_instances(
                        old_sub=old_sub,
                        new_sub=new_sub,
                        connectivity=ext,
                        sym_pairs=list(zip(np.nonzero(new_sym)))
                    )
                # if structure does not exist in pattern
                if structure_id == -1:
                    continue

                new_sub.set_id(structure_id)

                # get additional required permutations for each subpattern
                # to account for removing some symmetries
                if len(del_sym) > 0:    # high chance maybe
                    permute_old = get_restricted_perms(
                        sym_pairs=sym_poset.subtree(
                            range(0, old_sub.n_vrtx)),
                        perms=old_sub.get_automorphisms()
                    )
                else:
                    permute_old = []

                costs = \
                    pattern_cost.get_extend_costs(
                        old_sub=old_sub,
                        new_sub=new_sub,
                        n_paired=sum(ext)
                    )
                dp_tracker[k][structure_id] = {
                    'costs': costs,
                    'obj': new_sub,
                    'sym': new_sym,
                    'calc_cost':
                        COST_SCALE['intersect'] * costs['join_intersect'] +
                        COST_SCALE['storage'] * costs['storage'] +
                        dp_info['calc_cost'],
                    'step': {
                        'type': hybrid_action.EXTEND,
                        'old_sub': old_sub,
                        'connectivity': ext,
                        'permute': permute_old,        # for joining multiple times
                        'add_sym': add_sym             # vid restrictions
                    }
                }

        # join two patterns
        for dp_info_0 in dp_tracker[k-1].values():
            # TODO: all previous ones, not just k-1 [*]
            for dp_info_1 in dp_tracker[k-1].values():
                sub_0 = dp_info_0['obj']
                sub_1 = dp_info_1['obj']

                # generate pairings that are not symmetric
                for n_paired in range(1, min(sub_0.n_vrtx, sub_1.n_vrtx)):
                    pairings_0 = sub_0.get_nonauto_combs(n_paired)
                    pairings_1 = sub_1.get_nonauto_combs(n_paired)

                    for (p0, p1) in product(pairings_0, pairings_1):
                        reindex = _get_merged_reindex(sub_0, sub_1, p0, p1)
                        try:
                            new_sub = \
                                get_merged_pattern(
                                    sub_0=sub_0, paired_0=p0, sym_0=dp_info_0['sym'],
                                    sub_1=sub_1, paired_1=p1, sym_1=dp_info_1['sym']
                                )
                        except AssertionError:      # different induced adj/sym
                            continue

                        # TODO: del_sym is messy (indexed by merged pattern) and never used
                        new_sym, add_sym, del_sym, sym_poset = \
                            get_merge_symmetry(
                                sub_0=sub_0, paired_0=p0, sym_0=dp_info_0['sym'],
                                sub_1=sub_1, paired_1=p1, sym_1=dp_info_1['sym'],
                                new_sub=new_sub,
                                reindex=reindex
                            )
                        structure_id, instances = \
                            pattern.get_merged_instances(
                                sub_0=sub_0, paired_0=p0,
                                sub_1=sub_1, paired_1=p1,
                                new_sub=new_sub,
                                reindex=reindex,
                                sym_pairs=list(zip(np.nonzero(new_sym)))
                            )
                        # if structure does not exist in pattern
                        if structure_id == -1:
                            continue

                        new_sub.set_id(structure_id)

                        # get additional required permutations for each subpattern
                        # to account for removing some symmetries
                        permute_0 = get_restricted_perms(
                            sym_pairs=sym_poset.subtree(
                                range(0, sub_0.n_vrtx)),
                            perms=sub_0.get_automorphisms()
                        )

                        permute_1 = get_restricted_perms(
                            sym_pairs=sym_poset.subtree(reindex),
                            perms=sub_1.get_automorphisms()
                        )

                        costs = \
                            pattern_cost.get_merge_costs(
                                sub_0=sub_0,
                                sub_1=sub_1,
                                new_sub=new_sub
                            )
                        dp_tracker[k][structure_id] = {
                            'costs': costs,
                            'obj': new_sub,
                            'sym': new_sym,
                            'calc_cost':
                                COST_SCALE['intersect'] * costs['join_comp'] +
                                COST_SCALE['storage'] * costs['storage'] +
                                dp_info_0['calc_cost'] +
                                dp_info_1['calc_cost'],     # TODO: [*]
                            'step': {
                                'type': hybrid_action.MERGE,
                                'sub_0': sub_0,
                                'sub_1': sub_1,
                                'paired_0': p0,
                                'paired_1': p1,
                                'permute_0': permute_0,        # for joining multiple times
                                'permute_1': permute_1,
                                'add_sym': add_sym             # vid restrictions
                            }
                        }
        for i, entry in enumerate(dp_tracker[k].values()):
            _logger(title=f'tracker update {k}({i})', **entry)

        # TODO: pick top 5


def _logger(lvl=0, title='', **kwargs):
    lvl_map = {
        0: 'INFO',
        1: 'WARN',
        2: 'ERRR'
    }

    if len(title) > 0:
        title_div = ': '
        pad_left = '-' * ((72 - len(title)) // 2)
        pad_right = '-' * (72 - len(title) - len(pad_left))
    else:
        title_div = ''
        pad_left = '-' * 37
        pad_right = '-' * 37

    divider = f'+{pad_left}{lvl_map[lvl]}{title_div}{title}{pad_right}+'
    print(divider)
    for k, v in kwargs.items():
        print(f'{{{k}}}: {v}')
    print(divider)


if __name__ == '__main__':
    if len(argv) < 2:
        print('format: [./pattern.py pattern]')
    else:
        start_time = time()
        pattern = argv[1]

        indptr, indices = read_data('./codegen/input_patterns', pattern)
        print(indptr, indices)

        start_time = time()
        pattern = Pattern(None, indptr, indices)
        plan_dp(pattern)