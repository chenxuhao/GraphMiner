#pragma once

//#include "context.cuh"
#include "../graph_gpu.h"
#include "embedding.cuh"
#include "../operations.cuh"

__device__ void printout_embedding(unsigned level, IndexT *emb) {
  printf("embedding[");
  for (unsigned i = 0; i < level; i ++) {
    printf("%d, ", emb[i]);
  }
  printf("%d]\n", emb[level]);
}

inline __device__ bool is_connected(IndexT a, IndexT b, GraphGPU graph) {
  if (graph.getOutDegree(a) == 0 || graph.getOutDegree(b) == 0) return false;
  IndexT key = a;
  IndexT search = b;
  if (graph.getOutDegree(a) < graph.getOutDegree(b)) {
    key = b;
    search = a;
  } 
  IndexT begin = graph.edge_begin(search);
  IndexT end = graph.edge_end(search);
  IndexT l = begin;
  IndexT r = end-1;
  while (r >= l) { 
    IndexT mid = l + (r - l) / 2; 
    IndexT value = graph.getEdgeDst(mid);
    if (value == key) return true;
    if (value < key) l = mid + 1; 
    else r = mid - 1; 
  } 
  return false;
}

inline __device__ bool is_connected_dag(IndexT key, IndexT search, GraphGPU graph) {
  if (graph.getOutDegree(search) == 0) return false;
  IndexT begin = graph.edge_begin(search);
  IndexT end = graph.edge_end(search);
  IndexT l = begin;
  IndexT r = end-1;
  while (r >= l) { 
    IndexT mid = l + (r - l) / 2; 
    IndexT value = graph.getEdgeDst(mid);
    if (value == key) return true;
    if (value < key) l = mid + 1; 
    else r = mid - 1; 
  } 
  return false;
}

inline __device__ bool is_vertexInduced_automorphism(unsigned n, IndexT *emb, unsigned idx, IndexT src, IndexT dst, GraphGPU g) {
  // the new vertex id should be larger than the first vertex id
  if (dst <= emb[0]) return true;
  // the new vertex should not already exist in the embedding
  for (unsigned i = 1; i < n; ++i)
    if (dst == emb[i]) return true;
  // the new vertex should not already be extended by any previous vertex in the embedding
  for (unsigned i = 0; i < idx; ++i)
    if (is_connected(emb[i], dst, g)) return true;
  // the new vertex id should be larger than any vertex id after its source vertex in the embedding
  for (unsigned i = idx+1; i < n; ++i)
    if (dst < emb[i]) return true;
  return false;
}

// count 3-motifs
inline __device__ unsigned find_3motif_pattern_id(unsigned idx, IndexT dst, IndexT* emb, GraphGPU g, unsigned pos = 0) {
  unsigned pid = 1; // 3-chain
  if (idx == 0) {
    if (is_connected(emb[1], dst, g)) pid = 0; // triangle
#ifdef USE_WEDGE
    //else if (max_size == 4) is_wedge[pos] = 1; // wedge; used for 4-motif
#endif
  }
  return pid;
}

// count 4-motifs
inline __device__ unsigned find_4motif_pattern_id(unsigned n, unsigned idx, IndexT dst, IndexT* emb, unsigned pattern, GraphGPU g, unsigned pos = 0) {
  unsigned pid = pattern;
  unsigned num_edges = 1;
  if (pid == 0) { // extending a triangle
    for (unsigned j = idx+1; j < n; j ++)
      if (is_connected(emb[j], dst, g)) num_edges ++;
    pid = num_edges + 2; // p3: tailed-triangle; p4: diamond; p5: 4-clique
  } else { // extending a 3-chain
    assert(pid == 1);
    bool connected[3];
    for (int i = 0; i < 3; i ++) connected[i] = false;
    connected[idx] = true;
    for (unsigned j = idx+1; j < n; j ++) {
      if (is_connected(emb[j], dst, g)) {
        num_edges ++;
        connected[j] = true;
      }
    }
    if (num_edges == 1) {
      pid = 0; // p0: 3-path
      unsigned center = 1;
#ifdef USE_WEDGE
      //if (is_wedge[pos]) center = 0;
#else
      center = is_connected(emb[1], emb[2], g) ? 1 : 0;
#endif
      if (idx == center) pid = 1; // p1: 3-star
    } else if (num_edges == 2) {
      pid = 2; // p2: 4-cycle
      unsigned center = 1;
#ifdef USE_WEDGE
      //if (is_wedge[pos]) center = 0;
#else
      center = is_connected(emb[1], emb[2], g) ? 1 : 0;
#endif
      if (connected[center]) pid = 3; // p3: tailed-triangle
    } else {
      pid = 4; // p4: diamond
    }
  }
  return pid;
}

inline __device__ bool is_all_connected_dag(IndexT dst, IndexT *emb, IndexT end, GraphGPU g) {
  bool all_connected = true;
  for(IndexT i = 0; i < end; ++i) {
    IndexT from = emb[i];
    if (!is_connected_dag(dst, from, g)) {
      all_connected = false;
      break;
    }
  }
  return all_connected;
}

