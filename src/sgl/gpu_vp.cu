// Copyright (c) 2020 MIT
// Author: Xuhao Chen
#include <cub/cub.cuh>
#include "pattern.hh"
#include "graph_gpu.h"
#include "operations.cuh"
#include "cuda_profiler_api.h"
#include "cuda_launch_config.hpp"

#define BLK_SZ BLOCK_SIZE
typedef cub::BlockReduce<AccType, BLK_SZ> BlockReduce;
#include "diamond_vertex_warp.cuh"
#include "rectangle_vertex_warp.cuh"
//#include "house_vertex_warp.cuh"
//#include "pentagon_vertex_warp.cuh"

void SglSolver(Graph &g, Pattern &p, uint64_t &total, int, int) {
  size_t memsize = print_device_info(0);
  vidType nv = g.num_vertices();
  eidType ne = g.num_edges();
  auto md = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";
  if (memsize < mem_graph) std::cout << "Memory allocation failed\n";

  GraphGPU gg(g);
  AccType h_total = 0, *d_total;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total, &h_total, sizeof(AccType), cudaMemcpyHostToDevice));
  int k = 4;
  if (p.is_house() || p.is_pentagon()) k = 5;

  size_t nthreads = BLK_SZ;
  size_t nwarps = BLK_SZ/WARP_SIZE;
  size_t ntasks = nv;
  size_t nblocks = (ntasks-1)/nwarps+1;
  if (nblocks > 65536) nblocks = 65536;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM;
  if (p.is_house()) {
    //max_blocks_per_SM = maximum_residency(house_warp_edge_nested, nthreads, 0);
  } else if (p.is_pentagon()) {
    //max_blocks_per_SM = maximum_residency(pentagon_warp_edge_nested, nthreads, 0);
  } else if (p.is_rectangle()) {
    max_blocks_per_SM = maximum_residency(rectangle_warp_vertex_nested, nthreads, 0);
  } else {
    max_blocks_per_SM = maximum_residency(diamond_warp_vertex_nested, nthreads, 0);
  }
  size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  if (p.is_diamond() || p.is_pentagon()) nblocks = std::min(8*max_blocks, nblocks);
  std::cout << p.get_name() << "_warp_vertex max_blocks_per_SM = " << max_blocks_per_SM << "\n";

  size_t per_block_vlist_size = nwarps * size_t(k-3) * size_t(md) * sizeof(vidType);
  if (k > 3) {
    int64_t nb = int64_t(float(memsize) - float(mem_graph)) / int64_t(per_block_vlist_size);
    if (nb < nblocks) nblocks = nb;
  }
  std::cout << "CUDA subgraph listing (" << nblocks << " CTAs, " << nthreads << " threads/CTA) ...\n";
  size_t list_size = nblocks * per_block_vlist_size;
  vidType *frontier_list; // each warp has (k-3) vertex sets; each set has size of max_degree
  if (p.is_rectangle()) list_size = 0;
  std::cout << "frontier list size " << list_size/(1024*1024) << " MB\n";
  if (list_size > 0) CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_list, list_size));

  Timer t;
  t.Start();
  cudaProfilerStart();
  if (p.is_house()) {
    //house_warp_vertex_nested<<<nblocks, nthreads>>>(ntasks, gg, frontier_list, md, d_total);
  } else if (p.is_pentagon()) {
    //pentagon_warp_vertex_nested<<<nblocks, nthreads>>>(ntasks, gg, d_total);
  } else if (p.is_rectangle()) {
    rectangle_warp_vertex_nested<<<nblocks, nthreads>>>(ntasks, gg, d_total);
  } else {
    diamond_warp_vertex_nested<<<nblocks, nthreads>>>(ntasks, gg, frontier_list, md, d_total);
  }
  cudaProfilerStop();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  std::cout << "runtime [cuda_vertex_parallel] = " << t.Seconds() << " sec\n";
  CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
  total = h_total;
  CUDA_SAFE_CALL(cudaFree(d_total));
}

