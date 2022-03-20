// Copyright (c) 2020 MIT
// Author: Xuhao Chen
#include <cub/cub.cuh>
#include "pattern.hh"
#include "cmap_gpu.h"
#include "graph_gpu.h"
#include "cuda_launch_config.hpp"
#define BLK_SZ BLOCK_SIZE
typedef cub::BlockReduce<AccType, BLK_SZ> BlockReduce;

#include "diamond_nested.cuh"
#include "rectangle_nested_cmap.cuh"
#include "house_edge_warp_nested.cuh"
#include "pentagon_edge_warp_nested.cuh"

void SglSolver(Graph &g, Pattern &p, uint64_t &total, int) {
  size_t memsize = print_device_info(0);
  vidType nv = g.num_vertices();
  eidType ne = g.num_edges();
  auto md = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";
  GraphGPU gg(g);
  size_t nnz = gg.init_edgelist(g, 1);
  std::cout << "Number of elements in the edgelist: " << nnz << "\n";

  AccType h_total = 0, *d_total;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total, &h_total, sizeof(AccType), cudaMemcpyHostToDevice));

  int k = 4;
  if (p.is_house() || p.is_pentagon()) k = 5;
  size_t nwarps = WARPS_PER_BLOCK;
  size_t per_block_vlist_size = nwarps * size_t(k-3) * size_t(md) * sizeof(vidType);
  size_t per_block_cmap_size = nwarps * size_t(nv) * sizeof(cmap_vt);
  size_t nthreads = BLK_SZ;
  size_t nblocks = (nnz-1)/WARPS_PER_BLOCK+1;
  if (nblocks > 65536) nblocks = 65536;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(rectangle_warp_edge_nested_cmap, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  nblocks = std::min(max_blocks, nblocks);
  if (k > 3) {
    size_t per_block_size = per_block_cmap_size + per_block_vlist_size;
    if (memsize*0.9 < mem_graph) {
      std::cout << "Memory allocation failed\n";
      exit(0);
    }
    size_t nb = (memsize*0.9 - mem_graph) / per_block_size;
    if (nb < nblocks) nblocks = nb;
  }
  std::cout << "CUDA subgraph listing (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  vidType *frontier_list; // each warp has (k-3) vertex sets; each set has size of max_degree
  size_t list_size = nblocks * per_block_vlist_size;
  if (!p.is_rectangle())
    CUDA_SAFE_CALL(cudaMalloc((void**)&frontier_list, list_size));
  size_t cmap_size = nblocks * per_block_cmap_size;
  std::cout << "cmap size " << cmap_size/(1024*1024) << " MB, vlist size " << list_size/(1024*1024) << " MB)\n";
  //cmap_vt* cmap;
  //CUDA_SAFE_CALL(cudaMalloc((void**)&cmap, cmap_size));
  //CUDA_SAFE_CALL(cudaMemset(cmap, 0, cmap_size));
  cmap_gpu cmap(NUM_BUCKETS, BUCKET_SIZE, nwarps*nblocks);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  Timer t;
  t.Start();
  if (p.is_house()) {
    house_warp_edge_nested<<<nblocks, nthreads>>>(nnz, gg, frontier_list, md, d_total);
  } else if (p.is_pentagon()) {
    pentagon_warp_edge_nested<<<nblocks, nthreads>>>(nnz, gg, d_total);
  } else if (p.is_rectangle()) {
    rectangle_warp_edge_nested_cmap<<<nblocks, nthreads>>>(nnz, nv, gg, cmap, d_total);
    //rectangle_warp_vertex_nested_cmap<<<nblocks, nthreads>>>(nv, md, gg, cmap, d_total);
  } else {
    diamond_warp_edge_nested<<<nblocks, nthreads>>>(nnz, gg, frontier_list, md, d_total);
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  std::cout << "runtime [cuda_cmap] = " << t.Seconds() << " sec\n";
  CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
  total = h_total;
  CUDA_SAFE_CALL(cudaFree(d_total));
}

