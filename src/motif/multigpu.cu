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
#include "motif3_edge_warp.cuh"
#include "motif4_edge_warp.cuh"
#include "motif4_edge_warp_fission.cuh"
#include <thread>

__global__ void clear(AccType *counts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  counts[i] = 0;
}

void MotifSolver(Graph &g, int k, std::vector<uint64_t> &accum, int n_gpus, int chunk_size) {
  int ndevices = 0;
  eidType nnz = 0;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&ndevices));
  size_t memsize = print_device_info(0);
  auto nv = g.num_vertices();
  auto ne = g.num_edges();
  auto md = g.get_max_degree();
  size_t npatterns = accum.size();
  nnz = g.init_edgelist();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  //std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";
  //std::cout << "Total edgelist size = " << nnz << " number of patterns = " << npatterns << "\n";
  if (memsize < mem_graph) std::cout << "Graph too large. Unified Memory (UM) allocation required\n";

  if (ndevices < n_gpus) {
    std::cout << "Only " << ndevices << " GPUs available\n";
  } else ndevices = n_gpus;

  // split the edgelist onto multiple gpus
  Timer t;
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

  std::vector<GraphGPU> d_graphs(ndevices);
  std::vector<std::thread> threads2;
  double graph_copy_time = 0, edgelist_copy_time = 0;
  t.Start();
  for (int i = 0; i < ndevices; i++) {
    threads2.push_back(std::thread([&,i]() {
    CUDA_SAFE_CALL(cudaSetDevice(i));
    d_graphs[i].init(g, i, ndevices);
    }));
  }
  for (auto &thread: threads2) thread.join();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  graph_copy_time = t.Seconds();
  std::cout << "Time on copying graph to GPUs: " << graph_copy_time << " sec\n";

  std::vector<std::thread> threads;
  t.Start();
  for (int i = 0; i < ndevices; i++) {
    threads.push_back(std::thread([&,i]() {
    CUDA_SAFE_CALL(cudaSetDevice(i));
#ifdef EVEN_SPLIT
    d_graphs[i].copy_edgelist_to_device(nnz, g);
#else
    d_graphs[i].copy_edgelist_to_device(num_tasks, src_ptrs, dst_ptrs);
#endif
    }));
  }
  for (auto &thread: threads) thread.join();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  edgelist_copy_time = t.Seconds();

  size_t nthreads = BLOCK_SIZE;
  size_t nwarps = WARPS_PER_BLOCK;
  size_t nblocks = (n_tasks_per_gpu-1)/WARPS_PER_BLOCK+1;
  size_t n_lists = 2; // for 4-motif
  if (k == 3) n_lists = 0;
  size_t per_block_vlist_size = nwarps * n_lists * size_t(md) * sizeof(vidType);
  if (nblocks > 65536) nblocks = 65536;
  if (k > 3) {
    size_t nb = (memsize*0.9 - mem_graph) / per_block_vlist_size;
    if (nb < nblocks) nblocks = nb;
  }
 
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(motif3_warp_edge, nthreads, 0);
  if (k==4) max_blocks_per_SM = maximum_residency(motif4_warp_edge, nthreads, 0);
  std::cout << k << "-motif max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  nblocks = std::min(6*max_blocks, nblocks); 
  std::cout << "CUDA " << k << "-motif listing/counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
  size_t list_size = nblocks * per_block_vlist_size;
  std::cout << "frontier list size: " << list_size/(1024*1024) << " MB\n";
  for (int i = 0; i < ndevices; i++)
    std::cout << "num_tasks[" << i << "] = " << num_tasks[i] << "\n";

  std::vector<AccType> h_counts(ndevices*npatterns, 0);
  std::vector<AccType *> d_counts(ndevices);
  std::vector<vidType *> frontier_list(ndevices);
  for (int i = 0; i < ndevices; i++) {
    CUDA_SAFE_CALL(cudaSetDevice(i));
    CUDA_SAFE_CALL(cudaMalloc(&d_counts[i], npatterns*sizeof(AccType)));
    clear<<<1, npatterns>>>(d_counts[i]);
    if (k==4) CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_list[i], list_size));
  }
  std::vector<Timer> subt(ndevices);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
 
  t.Start();
  std::vector<std::thread> threads1;
  for (int i = 0; i < ndevices; i++) {
    threads1.push_back(std::thread([&,i]() {
    cudaSetDevice(i);
    subt[i].Start();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //CUDA_SAFE_CALL(cudaMemcpy(d_counts[i], &h_counts[i*npatterns], npatterns*sizeof(AccType), cudaMemcpyHostToDevice));
    if (k == 3) {
      motif3_warp_edge<<<nblocks, nthreads>>>(num_tasks[i], d_graphs[i], frontier_list[i], md, d_counts[i]);
    } else if (k == 4) {
#ifdef FISSION
      star3_warp_edge<<<nblocks, nthreads>>>(num_tasks[i], d_graphs[i], frontier_list[i], md, d_counts[i]);
      path4_warp_edge<<<nblocks, nthreads>>>(num_tasks[i], d_graphs[i], frontier_list[i], md, d_counts[i]);
      cycle4_warp_edge<<<nblocks, nthreads>>>(num_tasks[i], d_graphs[i], frontier_list[i], md, d_counts[i]);
      motif4_triangle<<<nblocks, nthreads>>>(num_tasks[i], d_graphs[i], frontier_list[i], md, d_counts[i]);
#else
      motif4_warp_edge<<<nblocks, nthreads>>>(num_tasks[i], d_graphs[i], frontier_list[i], md, d_counts[i]);
#endif
    } else {
      std::cout << "Not supported right now\n";
    }
    CUDA_SAFE_CALL(cudaMemcpy(&h_counts[i*npatterns], d_counts[i], npatterns*sizeof(AccType), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    subt[i].Stop();
    }));
  }
  for (auto &thread: threads1) thread.join();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  for (size_t i = 0; i < npatterns; i ++) {
    for (int j = 0; j < ndevices; j++) {
      accum[i] += h_counts[j*npatterns+i];
    }
  }
  t.Stop();
  for (int i = 0; i < ndevices; i++)
    std::cout << "runtime[gpu" << i << "] = " << subt[i].Seconds() <<  " sec\n";
  std::cout << "runtime = " << t.Seconds() <<  " sec\n";
  std::cout << "Time on splitting edgelist to GPUs: " << split_time << " sec\n";
  std::cout << "Time on copy graph to GPU: " << graph_copy_time <<  " sec\n";
  std::cout << "Time on copy edgelist to GPU: " << edgelist_copy_time <<  " sec\n";
}

