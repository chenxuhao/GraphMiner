// Copyright (c) 2020 MIT
// Author: Xuhao Chen
#include <cub/cub.cuh>
#include "edgelist.h"
#include "graph_gpu.h"
#include "scheduler.h"
#include "operations.cuh"
#include "cuda_launch_config.hpp"
//#define EVEN_SPLIT
#define BLK_SZ BLOCK_SIZE

typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;
#include "clique4_warp_edge.cuh"
#include "clique5_warp_edge.cuh"
#include "clique6_warp_edge.cuh"
#include "clique7_warp_edge.cuh"
#include "clique8_warp_edge.cuh"
#include <thread>

void CliqueSolver(Graph &g, int k, uint64_t &total, int n_gpus, int chunk_size) {
  int ndevices = 0;
  eidType nnz = 0;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&ndevices));
  size_t memsize = print_device_info(0);
  auto nv = g.num_vertices();
  auto ne = g.num_edges();
  auto md = g.get_max_degree();
  nnz = g.init_edgelist();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";
  std::cout << "Total edgelist size = " << nnz << "\n";
  if (memsize < mem_graph) std::cout << "Graph too large. Unified Memory (UM) required\n";

  if (ndevices < n_gpus) {
    std::cout << "Only " << ndevices << " GPUs available\n";
  } else ndevices = n_gpus;

  // split the edgelist onto multiple gpus
  Timer t;
  t.Start();
  eidType n_tasks_per_gpu = eidType(nnz-1) / eidType(ndevices) + 1;
#ifdef EVEN_SPLIT
  std::vector<eidType> num_tasks(ndevices, n_tasks_per_gpu);
  num_tasks[ndevices-1] = nnz - (ndevices-1) * n_tasks_per_gpu;
#else
  std::vector<vidType*> src_ptrs, dst_ptrs;
  Scheduler scheduler;
  //auto num_tasks = scheduler.least_first(ndevices, g, src_ptrs, dst_ptrs, chunk_size);
  auto num_tasks = scheduler.round_robin(ndevices, g, src_ptrs, dst_ptrs, chunk_size);
  //auto num_tasks = scheduler.vertex_chunking(ndevices, g, src_ptrs, dst_ptrs, chunk_size);
#endif
  t.Stop();
  for (int i = 0; i < ndevices; i++)
    std::cout << "GPU[" << i << "] is assigned " << num_tasks[i] << " tasks\n";
  std::cout << "Time on splitting the tasks = " << t.Seconds() <<  " sec\n";

  std::vector<GraphGPU> d_graphs(ndevices);
  t.Start();
  for (int i = 0; i < ndevices; i++) {
    CUDA_SAFE_CALL(cudaSetDevice(i));
    d_graphs[i].init(g, i, ndevices);
#ifdef EVEN_SPLIT
    d_graphs[i].copy_edgelist_to_device(nnz, g);
#else
    d_graphs[i].copy_edgelist_to_device(num_tasks, src_ptrs, dst_ptrs);
#endif
  }
  t.Stop();
  std::cout << "Total GPU copy time (graph+edgelist) = " << t.Seconds() <<  " sec\n";

  size_t nthreads = BLOCK_SIZE;
  size_t nwarps = WARPS_PER_BLOCK;
  size_t nblocks = (n_tasks_per_gpu-1)/WARPS_PER_BLOCK+1;
  size_t per_block_vlist_size = nwarps * size_t(k-3) * size_t(md) * sizeof(vidType);
  if (nblocks > 65536) nblocks = 65536;
  if (memsize > mem_graph) {
    size_t nb = (memsize - mem_graph) / per_block_vlist_size;
    if (nb < nblocks) nblocks = nb;
  }

  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(clique4_warp_edge, nthreads, 0);
  if (k==5) max_blocks_per_SM = maximum_residency(clique5_warp_edge, nthreads, 0);
  if (k==6) max_blocks_per_SM = maximum_residency(clique6_warp_edge, nthreads, 0);
  if (k==7) max_blocks_per_SM = maximum_residency(clique7_warp_edge, nthreads, 0);
  if (k==8) max_blocks_per_SM = maximum_residency(clique8_warp_edge, nthreads, 0);
  std::cout << k << "-clique max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  nblocks = std::min(6*max_blocks, nblocks); 
  std::cout << "CUDA " << k << "-clique listing/counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
  size_t list_size = nblocks * per_block_vlist_size;
  std::cout << "frontier list size: " << list_size/(1024*1024) << " MB\n";

  std::vector<AccType> h_counts(ndevices, 0);
  std::vector<AccType *> d_count(ndevices);
  std::vector<vidType *> frontier_list(ndevices);
  for (int i = 0; i < ndevices; i++) {
    CUDA_SAFE_CALL(cudaSetDevice(i));
    CUDA_SAFE_CALL(cudaMalloc(&d_count[i], sizeof(AccType)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_list[i], list_size));
  }
  std::vector<std::thread> threads;
  std::vector<Timer> subt(ndevices);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
 
  t.Start();
  for (int i = 0; i < ndevices; i++) {
    threads.push_back(std::thread([&,i]() {
    cudaSetDevice(i);
    subt[i].Start();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(d_count[i], &h_counts[i], sizeof(AccType), cudaMemcpyHostToDevice));
    if (k == 4) {
      clique4_warp_edge<<<nblocks, nthreads>>>(num_tasks[i], d_graphs[i], frontier_list[i], md, d_count[i]);
    } else if (k == 5) {
      clique5_warp_edge<<<nblocks, nthreads>>>(num_tasks[i], d_graphs[i], frontier_list[i], md, d_count[i]);
    } else if (k == 6) {
      clique6_warp_edge<<<nblocks, nthreads>>>(num_tasks[i], d_graphs[i], frontier_list[i], md, d_count[i]);
    } else if (k == 7) {
      clique7_warp_edge<<<nblocks, nthreads>>>(num_tasks[i], d_graphs[i], frontier_list[i], md, d_count[i]);
    } else if (k == 8) {
      clique8_warp_edge<<<nblocks, nthreads>>>(num_tasks[i], d_graphs[i], frontier_list[i], md, d_count[i]);
    } else {
      std::cout << "Not supported right now\n";
    }
    CUDA_SAFE_CALL(cudaMemcpy(&h_counts[i], d_count[i], sizeof(AccType), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    subt[i].Stop();
    }));
  }
  for (auto &thread: threads) thread.join();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  for (int i = 0; i < ndevices; i++) total += h_counts[i];
  t.Stop();
  for (int i = 0; i < ndevices; i++)
    std::cout << "runtime[gpu" << i << "] = " << subt[i].Seconds() <<  " sec\n";
  std::cout << "runtime = " << t.Seconds() <<  " sec\n";
}

