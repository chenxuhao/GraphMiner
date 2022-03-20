// Copyright (c) 2020 MIT
// Author: Xuhao Chen
#include <cub/cub.cuh>
#include "pattern.hh"
#include "edgelist.h"
#include "graph_gpu.h"
#include "scheduler.h"
#include "operations.cuh"
#include "cuda_launch_config.hpp"
#define BLK_SZ BLOCK_SIZE
typedef cub::BlockReduce<AccType, BLK_SZ> BlockReduce;

//#include "diamond_iterative.cuh"
#include "diamond_nested.cuh"
//#include "diamond_count.cuh"
//#include "rectangle_iterative.cuh"
#include "rectangle_nested.cuh"
//#include "rectangle_nested_balanced.cuh"
#include "house_edge_warp_nested.cuh"
#include "pentagon_edge_warp_nested.cuh"

//#define EVEN_SPLIT
#include <thread>
void SglSolver(Graph &g, Pattern &p, uint64_t &total, int n_gpus, int chunk_size) {
  int ndevices = 0;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&ndevices));
  size_t memsize = print_device_info(0);
  auto nv = g.num_vertices();
  auto ne = g.num_edges();
  auto md = g.get_max_degree();
  bool edge_sym_break = true; // TODO: this is pattern specific
  auto nnz = g.init_edgelist(edge_sym_break);
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  //std::cout << "\nGPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";
  std::cout << "Total edgelist size = " << nnz << "\n";
  Timer t;

  int k = 4;
  if (p.is_house() || p.is_pentagon()) k = 5;
  if (ndevices < n_gpus) {
    std::cout << "Only " << ndevices << " GPUs available\n";
  } else ndevices = n_gpus;
 
  // split the edgelist onto multiple gpus
  double split_time = 0;
  t.Start();
  eidType n_tasks_per_gpu = eidType(nnz-1) / eidType(ndevices) + 1;
#ifdef EVEN_SPLIT
  std::vector<eidType> num_tasks(ndevices, n_tasks_per_gpu);
  num_tasks[ndevices-1] = nnz - (ndevices-1) * n_tasks_per_gpu;
#else
  std::vector<vidType*> src_ptrs, dst_ptrs;
  Scheduler scheduler;
  //auto num_tasks = scheduler.split_edgelist(ndevices, g, src_ptrs, dst_ptrs, chunk_size);
  auto num_tasks = scheduler.round_robin(ndevices, g, src_ptrs, dst_ptrs, chunk_size);
#endif
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  split_time = t.Seconds();
  std::cout << "Time on splitting edgelist to GPUs: " << split_time << " sec\n";

  std::vector<std::thread> threads;
  std::vector<GraphGPU> d_graphs(ndevices);
  double copy_time = 0;
  t.Start();
  for (int i = 0; i < ndevices; i++) {
    threads.push_back(std::thread([&,i]() {
    CUDA_SAFE_CALL(cudaSetDevice(i));
    d_graphs[i].init(g, i, ndevices);
#ifdef EVEN_SPLIT
    d_graphs[i].copy_edgelist_to_device(nnz, g, edge_sym_break);
#else
    d_graphs[i].copy_edgelist_to_device(num_tasks, src_ptrs, dst_ptrs);
#endif
    }));
  }
  for (auto &thread: threads) thread.join();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  copy_time = t.Seconds();

  size_t nthreads = BLK_SZ;
  size_t nblocks = (n_tasks_per_gpu-1)/WARPS_PER_BLOCK+1;
  if (nblocks > 65536) nblocks = 65536;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(rectangle_warp_edge_nested, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  nblocks = std::min(6*max_blocks, nblocks); 
  std::cout << "CUDA subgraph listing (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  std::vector<AccType> h_counts(ndevices, 0);
  std::vector<AccType *> d_count(ndevices);
  for (int i = 0; i < ndevices; i++) {
    CUDA_SAFE_CALL(cudaSetDevice(i));
    CUDA_SAFE_CALL(cudaMalloc(&d_count[i], sizeof(AccType)));
  }
  std::vector<Timer> subt(ndevices);
  std::vector<std::thread> threads1;

  t.Start();
  for (int i = 0; i < ndevices; i++) {
    threads1.push_back(std::thread([&,i]() {
    CUDA_SAFE_CALL(cudaSetDevice(i));
    subt[i].Start();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_count[i], sizeof(AccType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_count[i], &h_counts[i], sizeof(AccType), cudaMemcpyHostToDevice));
    if (p.is_house()) {
      //house_warp_edge_nested<<<nblocks, nthreads>>>(num_tasks[i], d_graphs[i], frontier_list, md, d_count[i]);
    } else if (p.is_pentagon()) {
      pentagon_warp_edge_nested<<<nblocks, nthreads>>>(num_tasks[i], d_graphs[i], d_count[i]);
    } else if (p.is_rectangle()) {
      rectangle_warp_edge_nested<<<nblocks, nthreads>>>(num_tasks[i], d_graphs[i], d_count[i]);
    } else {
      //diamond_warp_edge_nested<<<nblocks, nthreads>>>(num_tasks[i], d_graphs[i], frontier_list, md, d_count[i]);
    }
    CUDA_SAFE_CALL(cudaMemcpy(&h_counts[i], d_count[i], sizeof(AccType), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    subt[i].Stop();
    }));
  }
  for (auto &thread: threads1) thread.join();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  for (int i = 0; i < ndevices; i++) total += h_counts[i];
  t.Stop();
  for (int i = 0; i < ndevices; i++)
    std::cout << "runtime[gpu" << i << "] = " << subt[i].Seconds() <<  " sec\n";
  std::cout << "Time on GPU execution, runtime = " << t.Seconds() <<  " sec\n";
  std::cout << "Time on splitting edgelist to GPUs: " << split_time << " sec\n";
  std::cout << "Time on data copy to GPU (graph+edgelist): " << copy_time <<  " sec\n";
}

