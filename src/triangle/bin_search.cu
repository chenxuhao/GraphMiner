// Copyright (c) 2020 MIT
// Author: Xuhao Chen
#include <cub/cub.cuh>
#include "timer.h"
//#include "edgelist.h"
#include "graph_gpu.h"
#include "operations.cuh"
#include "cuda_launch_config.hpp"

typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;
#include "bin_search.cuh"

//#define USE_CTA
void TCSolver(Graph &g, uint64_t &total, int, int) {
  size_t memsize = print_device_info(0);
  auto nv = g.num_vertices();
  auto ne = g.num_edges();
  auto md = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";

  GraphGPU gg(g);
  auto nnz = gg.init_edgelist(g);
  std::cout << "Edge parallel: edgelist size = " << nnz << "\n";
  size_t nthreads = BLOCK_SIZE;
#ifdef USE_CTA
  size_t nblocks = ne;
#else
  size_t nblocks = (ne-1)/WARPS_PER_BLOCK+1;
#endif
  if (nblocks > 65536) nblocks = 65536;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(warp_edge, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  //size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  //nblocks = std::min(max_blocks, nblocks);
  std::cout << "CUDA triangle counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
 
  AccType h_total = 0, *d_total;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total, &h_total, sizeof(AccType), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  Timer t;
  t.Start();
#ifdef USE_CTA
  cta_edge<<<nblocks, nthreads>>>(ne, gg, d_total);
#else
  warp_edge<<<nblocks, nthreads>>>(ne, gg, d_total);
#endif
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  std::cout << "runtime [gpu_bs] = " << t.Seconds() << " sec\n";
  std::cout << "throughput = " << double(nnz) / t.Seconds() / 1e9 << " billion Traversed Edges Per Second (TEPS)\n";
  CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
  total = h_total;
  CUDA_SAFE_CALL(cudaFree(d_total));
}

