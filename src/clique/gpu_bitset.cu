// Copyright (c) 2020 MIT
// Author: Xuhao Chen
#include <cub/cub.cuh>
#include "bitsets.h"
#include "graph_gpu.h"
#include "operations.cuh"
#include "cuda_launch_config.hpp"

#define BLK_SZ BLOCK_SIZE
typedef cub::BlockReduce<AccType, BLK_SZ> BlockReduce;
#include "clique4_warp_edge_bitset.cuh"
#include "clique5_warp_edge_bitset.cuh"
#include "clique6_warp_edge.cuh"
#include "clique7_warp_edge.cuh"
#include "clique8_warp_edge.cuh"

void CliqueSolver(Graph &g, int k, uint64_t &total, int, int) {
  assert(k > 3);
  size_t memsize = print_device_info(0);
  vidType nv = g.num_vertices();
  eidType ne = g.num_edges();
  auto md = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";
  if (memsize < mem_graph) std::cout << "Graph too large. Unified Memory (UM) required\n";

  GraphGPU gg(g);
  gg.init_edgelist(g);
  size_t nwarps = WARPS_PER_BLOCK;
  size_t nthreads = BLK_SZ;
  size_t nblocks = (ne-1)/WARPS_PER_BLOCK+1;
  size_t per_block_vlist_size = nwarps * size_t(k-3) * size_t(md) * sizeof(vidType);
  if (nblocks > 65536) nblocks = 65536;
  size_t nb = (memsize - mem_graph) / per_block_vlist_size;
  if (nb < nblocks) nblocks = nb;

  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(clique4_warp_edge_bitset, nthreads, 0);
  if (k==5) max_blocks_per_SM = maximum_residency(clique5_warp_edge_bitset, nthreads, 0);
  if (k==6) max_blocks_per_SM = maximum_residency(clique6_warp_edge, nthreads, 0);
  if (k==7) max_blocks_per_SM = maximum_residency(clique7_warp_edge, nthreads, 0);
  if (k==8) max_blocks_per_SM = maximum_residency(clique8_warp_edge, nthreads, 0);
  std::cout << k << "-clique max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  nblocks = std::min(max_blocks, nblocks);
  std::cout << "CUDA " << k << "-clique listing/counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
  size_t list_size = nblocks * per_block_vlist_size;
  std::cout << "frontier list size: " << list_size/(1024*1024) << " MB\n";
  vidType *frontier_list; // each warp has (k-3) vertex sets; each set has size of max_degree
  CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_list, list_size));

  MultiBitsets<> adj_lists(nblocks*nwarps, md, md);	
  AccType h_total = 0, *d_total;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total, &h_total, sizeof(AccType), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  Timer t;
  t.Start();
  if (k == 4) {
    clique4_warp_edge_bitset<<<nblocks, nthreads>>>(ne, gg, frontier_list, adj_lists, md, d_total);
  } else if (k == 5) {
    clique5_warp_edge_bitset<<<nblocks, nthreads>>>(ne, gg, frontier_list, adj_lists, md, d_total);
  } else if (k == 6) {
    clique6_warp_edge<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total);
  } else if (k == 7) {
    clique7_warp_edge<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total);
  } else if (k == 8) {
    clique8_warp_edge<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total);
  } else {
    std::cout << "Not supported right now\n";
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  std::cout << "runtime [gpu] = " << t.Seconds() << " sec\n";
  CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
  total = h_total;
  CUDA_SAFE_CALL(cudaFree(d_total));
}

