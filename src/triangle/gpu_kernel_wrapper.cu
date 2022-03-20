#include <thread>
#include <cub/cub.cuh>
#include "graph_gpu.h"
#include "edgelist.h"
#include "cuda_launch_config.hpp"

typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;
#include "bs_warp_edge.cuh"

void set_nblocks(size_t nthreads, size_t &nblocks);
std::vector<eidType> even_task_split(int ngpus, int nranks, int rank, Graph &g, std::vector<GraphGPU> &dgs);

void triangle_warp_edge(int n_gpus, int nranks, int rank, Graph &g, uint64_t &total) {
  std::string head = "[Host " + std::to_string(rank) + "] ";
  auto nv = g.num_vertices();
  auto ne = g.num_edges();
  //auto md = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  size_t memsize = print_device_info(0, 1);
  std::cout << head << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";

  int ndevices = 0;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&ndevices));
  if (ndevices < n_gpus) {
    std::cout << "Only " << ndevices << " GPUs available\n";
    exit(0);
  } else ndevices = n_gpus;

  std::vector<GraphGPU> d_graphs;
  auto num_tasks = even_task_split(ndevices, nranks, rank, g, d_graphs);

  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = (num_tasks[0]-1)/WARPS_PER_BLOCK+1;
  set_nblocks(nthreads, nblocks);
  std::cout << head << "CUDA TC (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
  std::vector<AccType *> d_count(ndevices);
  for (int i = 0; i < ndevices; i++) {
    CUDA_SAFE_CALL(cudaSetDevice(i));
    CUDA_SAFE_CALL(cudaMalloc(&d_count[i], sizeof(AccType)));
  }
  std::vector<std::thread> threads;
  std::vector<Timer> subt(ndevices);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  Timer t; 
  t.Start();
  std::vector<AccType> h_counts(ndevices, 0);
  for (int i = 0; i < ndevices; i++) {
    threads.push_back(std::thread([&,i]() {
    cudaSetDevice(i);
    //std::cout << head << "GPU_" << i << " num_tasks: " << num_tasks[i] << "\n";
    subt[i].Start();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(d_count[i], &h_counts[i], sizeof(AccType), cudaMemcpyHostToDevice));
    warp_edge<<<nblocks, nthreads>>>(num_tasks[i], d_graphs[i], d_count[i]);
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
    std::cout << head << "runtime [gpu" << i << "] = " << subt[i].Seconds() <<  " sec\n";
  //for (int i = 0; i < ndevices; i++)
  //  std::cout << head << "Local sum [gpu" << i << "] = " << h_counts[i] << "\n";
  std::cout << head << "runtime [dist_gpu] = " << t.Seconds() << " sec\n";
  //std::cout << head << "Local sum total = " << total << "\n";
}

void set_nblocks(size_t nthreads, size_t &nblocks) {
  if (nblocks > 65536 || nblocks == 0) nblocks = 65536;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(warp_edge, nthreads, 0);
  //std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  nblocks = std::min(6*max_blocks, nblocks); 
}

std::vector<eidType> even_task_split(int ngpus, int nranks, int rank, Graph &g, std::vector<GraphGPU> &d_graphs) {
  d_graphs.resize(ngpus); // allocate a device graph per GPU
  auto ne = g.num_edges();
  int ndevices = d_graphs.size();
  std::string head = "[Host " + std::to_string(rank) + "] ";

  // split the edgelist onto multiple gpus
  vidType ntasks_per_rank = ne / nranks;
  vidType begin = ntasks_per_rank * rank;
  vidType end = ntasks_per_rank * (rank+1);
  if (end > ne) end = ne;
  std::cout << head << ": [" << begin << ", " << end << ")\n";
  eidType n_tasks_per_gpu = eidType(end-begin-1) / eidType(ndevices) + 1;
  std::vector<eidType> num_tasks(ndevices, n_tasks_per_gpu);
  num_tasks[ndevices-1] = (end-begin) - (ndevices-1) * n_tasks_per_gpu;

  Timer t;
  t.Start();
  for (int i = 0; i < ndevices; i++) {
    CUDA_SAFE_CALL(cudaSetDevice(i));
    d_graphs[i].init(g, i, ndevices);
    std::cout << head << "[gpu" << i << "]: begin " << begin << " end " << end << "\n";
    d_graphs[i].copy_edgelist_to_device(begin, end, g);
  }
  t.Stop();
  std::cout << head << "Total GPU copy time (graph+edgelist) = " << t.Seconds() <<  " sec\n";
  return num_tasks;
}

std::vector<eidType> chunked_task_split(int nranks, int rank, int chunk_size, Graph &g, std::vector<GraphGPU> &d_graphs) {
  auto ne = g.num_edges();
  int ndevices = d_graphs.size();
  std::vector<vidType*> src_ptrs, dst_ptrs;
  auto num_tasks = g.split_edgelist(ndevices, src_ptrs, dst_ptrs, chunk_size);
  Timer t;
  t.Start();
  for (int i = 0; i < ndevices; i++) {
    CUDA_SAFE_CALL(cudaSetDevice(i));
    d_graphs[i].init(g, i, ndevices);
    d_graphs[i].copy_edgelist_to_device(num_tasks, src_ptrs, dst_ptrs);
  }
  t.Stop();
  std::cout << "Total GPU copy time (graph+edgelist) = " << t.Seconds() <<  " sec\n";
  return num_tasks;
}
