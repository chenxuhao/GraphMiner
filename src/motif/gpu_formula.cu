// Copyright (c) 2020 MIT
// Author: Xuhao Chen
#include <cub/cub.cuh>
#include "graph_gpu.h"
#include "operations.cuh"
#include "cuda_launch_config.hpp"
#define BLK_SZ BLOCK_SIZE

typedef cub::BlockReduce<AccType, BLK_SZ> BlockReduce;

#include "cycle4_clique4.cuh"
#include "cycle4_edge_warp.cuh"
#include "clique4_edge_warp.cuh"
#include "motif4_rest.cuh"
#include "motif3_formula_edge_warp.cuh"

__global__ void clear(AccType *accumulators) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  accumulators[i] = 0;
}

void MotifSolver(Graph &g, int k, std::vector<uint64_t> &accum, int, int) {
  size_t memsize = print_device_info(0);
  assert(k >= 3);
  vidType nv = g.num_vertices();
  eidType ne = g.num_edges();
  auto md = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";
  //if (memsize < mem_graph) { std::cout << "Memory allocation failed\n"; exit(0); }
  if (memsize < mem_graph) std::cout << "Graph too large. Unified Memory (UM) allocation required\n";

  GraphGPU gg(g);
  gg.init_edgelist(g);
  size_t npatterns = accum.size();
  AccType *h_accumulators = (AccType *)malloc(sizeof(AccType) * npatterns);
  for (int i = 0; i < npatterns; i++) h_accumulators[i] = 0;
  AccType *d_accumulators;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_accumulators, sizeof(AccType) * npatterns));
  clear<<<1, npatterns>>>(d_accumulators);
 
  size_t nwarps = WARPS_PER_BLOCK;
  size_t n_lists = 2;
  if (k == 3) n_lists = 0;
  size_t per_block_vlist_size = nwarps * n_lists * size_t(md) * sizeof(vidType);
  size_t nthreads = BLK_SZ;
  size_t nblocks = (ne-1)/WARPS_PER_BLOCK+1;
  if (nblocks > 65536) nblocks = 65536;
  if (k > 3) {
    size_t nb = (memsize*0.9 - mem_graph) / per_block_vlist_size;
    if (nb < nblocks) nblocks = nb;
  }
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM;
  max_blocks_per_SM = maximum_residency(motif4_rest, nthreads, 0);
  std::cout << "4-motif-rest: max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  if (k == 3)
    max_blocks_per_SM = maximum_residency(motif3_formula_warp_vertex, nthreads, 0);
  else {
    max_blocks_per_SM = maximum_residency(cycle4_warp_edge, nthreads, 0);
  }
  std::cout << k << "-motif: max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  //size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  //nblocks = std::min(max_blocks, nblocks);
  std::cout << "CUDA " << k << "-motif counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
  size_t list_size = nblocks * per_block_vlist_size;
  std::cout << "frontier list size: " << list_size/(1024*1024) << " MB\n";
  vidType *frontier_list; // each warp has (k-3) vertex sets; each set has size of max_degree
  CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_list, list_size));

  Timer t;
  t.Start();
  if (k == 3) {
    motif3_formula_warp_vertex<<<nblocks, nthreads>>>(nv, gg, d_accumulators);
  } else if (k == 4) {
    motif4_rest<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_accumulators);
    //cycle4_clique4<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_accumulators);
    cycle4_warp_edge<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_accumulators);
    clique4_warp_edge<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_accumulators);
  } else {
    std::cout << "Not supported right now\n";
  }
  CUDA_SAFE_CALL(cudaMemcpy(h_accumulators, d_accumulators, sizeof(AccType) * npatterns, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < npatterns; i ++) accum[i] = h_accumulators[i];
  if (k == 3) {
    accum[1] = accum[1]/2 - 3 * accum[0];
  } else if (k == 4) {
    accum[4] = accum[4] / 2 - accum[5] * 6;
    accum[2] = accum[2] / 2 - accum[4] * 2;
    accum[1] = accum[1] - accum[3] * 4;
    accum[0] = accum[0] / 6 - accum[2] / 3;
  } else {
    exit(1);
  }
  t.Stop();

  std::cout << "runtime [cuda_base] = " << t.Seconds() << " sec\n";
  CUDA_SAFE_CALL(cudaFree(d_accumulators));
}

