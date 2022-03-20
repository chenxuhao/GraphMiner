// Copyright (c) 2020 MIT
// Author: Xuhao Chen
#include <cub/cub.cuh>
#include "pattern.hh"
#include "graph_gpu.h"
#include "operations.cuh"
#include "cuda_profiler_api.h"
#include "cuda_launch_config.hpp"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <moderngpu/kernel_segsort.hxx>

#define BLK_SZ BLOCK_SIZE
#define ITEMS_PER_THREAD 16
typedef cub::BlockReduce<AccType, BLK_SZ> BlockReduce;
typedef cub::BlockRadixSort<int, BLK_SZ, ITEMS_PER_THREAD> BlockRadixSort;
typedef cub::BlockLoad<int, BLK_SZ, ITEMS_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE>   BlockLoad;
typedef cub::BlockStore<int, BLK_SZ, ITEMS_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE> BlockStore;
#include "rectangle_bj.cuh"

void SglSolver(Graph &g, Pattern &p, uint64_t &total, int, int) {
  size_t memsize = print_device_info(0);
  vidType nv = g.num_vertices();
  eidType ne = g.num_edges();
  auto md = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";
  if (memsize < mem_graph) std::cout << "Graph too large. Unified Memory (UM) required\n";

  GraphGPU gg(g);
  AccType h_total = 0, *d_total;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType)));
  int k = 4;
  if (p.is_house() || p.is_pentagon()) k = 5;

  int nthreads = BLK_SZ;
  int nwarps = BLK_SZ/WARP_SIZE;
  size_t ntasks = nv;
  //ntasks = gg.init_edgelist(g, 1);
  //std::cout << "Edge parallel: edgelist size = " << ntasks << "\n";
  int nblocks = (ntasks-1)/nwarps+1;
  if (nblocks > 65536) nblocks = 65536;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM;
  max_blocks_per_SM = maximum_residency(wedge_listing, nthreads, 0);
  int max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  nblocks = std::min(max_blocks, nblocks);
  std::cout << p.get_name() << " max_blocks_per_SM = " << max_blocks_per_SM << ", max_blocks = " << max_blocks << "\n";

  const size_t length = size_t(md) * size_t(md); // size of buffer for each warp
  size_t per_block_vlist_size = nwarps * 2 * length * sizeof(vidType);
  if (k > 3) {
    auto nb = int64_t(float(memsize)*0.9 - float(mem_graph)) / int64_t(per_block_vlist_size);
    if (nb < nblocks) nblocks = nb;
  }
  std::cout << "CUDA subgraph listing (" << nblocks << " CTAs, " << nthreads << " threads/CTA) ...\n";
  int total_warps = int(nblocks * nwarps); // total number of warps
  std::cout << "total_warps = " << total_warps << "\n";
  const size_t buffer_size = total_warps * length * sizeof(vidType); // total size of all buffers
  std::cout << "buffer size = " << buffer_size / (1024*1024) << " MB\n";

  vidType *buffer1, *buffer2; // each warp has 2 vertex sets; each set has size of max_degree
  //CUDA_SAFE_CALL(cudaMalloc((void **)&buffer1, buffer_size));
  //CUDA_SAFE_CALL(cudaMalloc((void **)&buffer2, buffer_size));
  std::vector<int> num_wedges(total_warps);
  int *d_num_wedges;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_num_wedges, total_warps*sizeof(int)));
  int *indices;
  CUDA_SAFE_CALL(cudaMalloc((void **)&indices, sizeof(int) * (total_warps+1)));
  mgpu::standard_context_t context;
  std::cout << "Start searching\n";

  Timer t;
  t.Start();
  cudaProfilerStart();
  for (vidType v = 0; v < nv; v += total_warps) { // each warp takes one vertex
    CUDA_SAFE_CALL(cudaMemset(d_num_wedges, 0, total_warps*sizeof(int)));
    wedge_counting<<<nblocks, nthreads>>>(v, nv, gg, d_num_wedges);
    thrust::exclusive_scan(thrust::device, d_num_wedges, d_num_wedges+total_warps+1, indices);
    int total_num_wedges = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&total_num_wedges, &indices[total_warps], sizeof(int), cudaMemcpyDeviceToHost));
    //std::cout << "Total number of wedges: " << total_num_wedges << "\n";
    CUDA_SAFE_CALL(cudaMalloc((void **)&buffer1, total_num_wedges*sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&buffer2, total_num_wedges*sizeof(vidType)));
    wedge_listing<<<nblocks, nthreads>>>(v, nv, gg, indices, buffer1, buffer2);
    //wedge_listing<<<nblocks, nthreads>>>(v, nv, gg, md, buffer1, buffer2, d_num_wedges);
    //CUDA_SAFE_CALL(cudaMemcpy(&num_wedges[0], d_num_wedges, sizeof(int)*total_warps, cudaMemcpyDeviceToHost));
    //std::cout << "Sorting for v from " << v << " to " << std::min(v+int(total_warps), nv) << "\n";
    //for (int wid = 0; wid < total_warps; wid++)
    //  thrust::sort_by_key(thrust::device, buffer2+wid*length, buffer2+wid*length+num_wedges[wid], buffer1+wid*length);
    mgpu::segmented_sort_indices(buffer2, buffer1, total_num_wedges, indices+1, total_warps, mgpu::less_t<int>(), context);
    h_total = 0;
    CUDA_SAFE_CALL(cudaMemcpy(d_total, &h_total, sizeof(AccType), cudaMemcpyHostToDevice));
    //enumerate_rectangles<<<nblocks, nthreads>>>(md, d_num_wedges, buffer1, buffer2, d_total);
    enumerate_rectangles<<<nblocks, nthreads>>>(indices, d_num_wedges, buffer1, buffer2, d_total);
    CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
    total += h_total;
  }
  cudaProfilerStop();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  std::cout << "runtime [cuda_base] = " << t.Seconds() << " sec\n";
  CUDA_SAFE_CALL(cudaFree(d_total));
}

