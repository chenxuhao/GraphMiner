
#include "query_plan.h"
#include <cub/cub.cuh>
#include "timer.h"
#include "graph_gpu.h"
#include "operations.cuh"
#include "cuda_profiler_api.h"
#include "cuda_launch_config.hpp"

__global__ void generateBN(int n, const PatternGPU qg, vidType* order, vidType* bns, vidType* bn_counts) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int m = qg.size();
  assert(m<16);
  bool visited_vertices[16];
  for (int i = 0; i < 16; i++) visited_vertices[i] = false;
  visited_vertices[order[0]] = true;
  vidType* bn = bns + tid*n;
  vidType* bn_count = bn_counts + tid*n;
  for (vidType i = 1; i < m; ++i) {
    auto v = order[i];
    auto nbrs_cnt = qg.get_degree(v);
    for (vidType j = 0; j < nbrs_cnt; ++j) {
      auto nbr = qg.get_neighbor(v, j);
      if (visited_vertices[nbr])
        bn[i*max_depth+bn_count[i]++] = nbr;
    }
    visited_vertices[v] = true;
  }
}

__device__ void generateValidCandidates(GraphGPU dg, int depth, vidType* embedding, bool* visited_vertices,
                                        vidType* bn, vidType* bn_cnt, vidType* order, 
                                        vidType* candidates, vidType* candidates_count, 
                                        vidType* idx_count, vidType* valid_candidate) {
  auto u = order[depth];
  idx_count[depth] = 0;
  for (vidType i = 0; i < candidates_count[u]; ++i) {
    auto v = candidates[u][i];
    if (!visited_vertices[v]) {
      bool valid = true;
      for (vidType j = 0; j < bn_cnt[depth]; ++j) {
        auto u_nbr = bn[depth][j];
        auto u_nbr_v = embedding[u_nbr];
        if (!dg->is_connected(v, u_nbr_v)) {
          valid = false;
          break;
        }
      }
      if (valid) {
        valid_candidate[depth][idx_count[depth]++] = v;
      }
    }
  }
}
/*
__device__ void generateValidCandidateIndex(int depth, vidType* idx_embedding, vidType* bn, vidType* bn_cnt, 
                                            Edges ***edge_matrix, vidType* order, int *temp_buffer,
                                            vidType* idx_count, int *valid_candidate_index) {
  auto u = order[depth];
  auto previous_bn = bn[depth][0];
  auto previous_index_id = idx_embedding[previous_bn];
  int valid_candidates_count = 0;
  Edges& previous_edge = *edge_matrix[previous_bn][u];
  valid_candidates_count = previous_edge.offset_[previous_index_id + 1] - previous_edge.offset_[previous_index_id];
  int* previous_candidates = previous_edge.edge_ + previous_edge.offset_[previous_index_id];
  memcpy(valid_candidate_index, previous_candidates, valid_candidates_count * sizeof(int));
  int temp_count;
  for (int i = 1; i < bn_cnt[depth]; ++i) {
    auto current_bn = bn[depth][i];
    Edges& current_edge = *edge_matrix[current_bn][u];
    auto current_index_id = idx_embedding[current_bn];
    auto current_candidates_count = current_edge.offset_[current_index_id + 1] - current_edge.offset_[current_index_id];
    int* current_candidates = current_edge.edge_ + current_edge.offset_[current_index_id];
    temp_count += intersect_num(current_candidates, current_candidates_count, valid_candidate_index, valid_candidates_count, temp_buffer);
    int *ptr = temp_buffer;
    temp_buffer = valid_candidate_index;
    valid_candidate_index = ptr;
    valid_candidates_count = temp_count;
  }
  idx_count[depth] = valid_candidates_count;
}
*/
__global__ void __launch_bounds__(BLOCK_SIZE, 8)
warp_GQL(eidType num, GraphGPU qg, vidType* idxes, vidType* bn_counts, vidType* idx_counts, 
         vidType*embeddings, vidType* idx_embeddings, vidType* bns, bool* masks, 
         int* valid_candidate_idxes, int *temp_buffers, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id   = thread_id   / WARP_SIZE;               // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  AccType count = 0;
 
  for (int tid = warp_id; tid < num; tid += num_warps) {
  //for (int i = 0; i < num; i++) {
    auto tid = omp_get_thread_num();
    auto &visited_vertices = masks[tid];
    auto &idx = idxes[tid];
    auto &bn = bns[tid];
    auto &bn_count = bn_counts[tid];
    auto &idx_count = idx_counts[tid];
    auto &embedding = embeddings[tid];
    auto &valid_candidate = valid_candidates[tid];
  
    auto v0 = candidates[start_vertex][i];;
    embedding[start_vertex] = v0;
    visited_vertices[v0] = true; // set mask
    idx[1] = 0;
    int cur_depth = 1;
    generateValidCandidates(dg, 1, embedding, visited_vertices, bn, bn_count, order,
                            candidates, candidates_count, idx_count, valid_candidate);
    while (1) { // Start DFS walk
      while (idx[cur_depth] < idx_count[cur_depth]) {
        auto u = order[cur_depth];
        auto v = valid_candidate[cur_depth][idx[cur_depth]];
        embedding[u] = v;
        visited_vertices[v] = true; // set mask
        idx[cur_depth] += 1;
        if (cur_depth == max_depth - 1) { // arrive the last level; backtrack
          embedding_cnt += 1;
          visited_vertices[v] = false; // resume mask
        } else {
          call_count += 1;
          cur_depth += 1;
          idx[cur_depth] = 0;
          generateValidCandidates(dg, cur_depth, embedding, visited_vertices, bn, bn_count,
                                  order, candidates, candidates_count, idx_count, valid_candidate);
        }
      }
      cur_depth -= 1;
      visited_vertices[embedding[order[cur_depth]]] = false; // resume masks
      if (cur_depth == 0) break;
    }
  }
  return embedding_cnt;
}
/*
__global__ void __launch_bounds__(BLOCK_SIZE, 8)
warp_LFTJ(eidType num, GraphGPU qg, vidType* idxes, vidType* bn_counts, vidType* idx_counts, 
          vidType*embeddings, vidType* idx_embeddings, vidType* bns, bool* masks, 
          int* valid_candidate_idxes, int *temp_buffers, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id   = thread_id   / WARP_SIZE;               // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  AccType count = 0;
 
  for (int tid = warp_id; tid < num; tid += num_warps) {
    auto &visited_vertices = masks[tid];
    auto &idx = idxes[tid];
    auto &bn = bns[tid];
    auto &bn_count = bn_counts[tid];
    auto &idx_count = idx_counts[tid];
    auto &embedding = embeddings[tid];
    auto &idx_embedding = idx_embeddings[tid];
    auto &temp_buffer = temp_buffers[tid];
    auto &valid_candidate_idx = valid_candidate_idxes[tid];
    auto v0 = candidates[start_vertex][0]; // data vertex at level 0
    embedding[start_vertex] = v0;
    visited_vertices[v0] = true;
    idx_embedding[start_vertex] = i;
    int cur_depth = 1;
    idx[1] = 0;
    generateValidCandidateIndex(1, idx_embedding, bn, bn_count, edge_matrix, order, 
                                temp_buffer, idx_count, valid_candidate_idx[1]);
    while (true) {
      while (idx[cur_depth] < idx_count[cur_depth]) {
        auto valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
        auto u = order[cur_depth];         // query vertex
        auto v = candidates[u][valid_idx]; // data vertex
        if (visited_vertices[v]) {
          idx[cur_depth] += 1;
          continue;
        }
        embedding[u] = v;
        idx_embedding[u] = valid_idx;
        visited_vertices[v] = true;
        idx[cur_depth] += 1;
        if (cur_depth == max_depth - 1) {
          embedding_cnt += 1;
          visited_vertices[v] = false;
        } else {
          call_count += 1;
          cur_depth += 1;
          idx[cur_depth] = 0;
          generateValidCandidateIndex(cur_depth, idx_embedding, bn, bn_count, edge_matrix, order, 
                                      temp_buffer, idx_count, valid_candidate_idx[cur_depth]);
        }
      }
      cur_depth -= 1;
      auto u = order[cur_depth];
      visited_vertices[embedding[u]] = false;
      if (cur_depth == 0) break;
    }
  }
}
*/
uint64_t QueryPlan::exploreGQL(const Graph *dg, const Pattern *qg, VertexLists &candidates, 
                               VertexList &candidates_count, VertexList &order, size_t &call_count) {
  size_t memsize = print_device_info(0);
  auto nv = dg->num_vertices();
  auto ne = dg->num_edges();
  auto md = dg->get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";

  int max_depth = qg->size();
  auto start_vertex = order[0];
  auto num = candidates_count[start_vertex];
  uint64_t embedding_cnt = 0;
  auto num = candidates_count[start_vertex]; // number of candidates at level 0 (i.e. #v0 for u0)

  size_t nthreads = BLOCK_SIZE;
  size_t nwarps = BLK_SZ/WARP_SIZE;
  size_t nblocks = (num-1)/WARPS_PER_BLOCK+1;
  if (nblocks > 65536) nblocks = 65536;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(warp_LFTJ, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  //size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  //nblocks = std::min(max_blocks, nblocks);
  std::cout << "CUDA triangle counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  for (int i = 0; i < nt; i++) {
    visited_query_vertices[start_vertex] = true;
    for (int j = 1; j < max_depth; ++j) {
      bns[i][j].resize(max_depth);
      auto v = order[j];
      for (auto u : qg->N(v)) {
        if (visited_query_vertices[u])
          bns[i][j][bn_counts[i][j]++] = u;
      }
      visited_query_vertices[v] = true;
    }
  }

  AccType h_total = 0, *d_total;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total, &h_total, sizeof(AccType), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  vidType* idxes, bn_counts, idx_counts, embeddings, idx_embeddings, bns, valid_candidates;
  bool* masks, visited_query_vertices;
  CUDA_SAFE_CALL(cudaMalloc((void **)&idxes, sizeof(vidType)*nwarps*max_depth));
  CUDA_SAFE_CALL(cudaMemset(idxes, 0, sizeof(vidType)*nwarps*max_depth));
  CUDA_SAFE_CALL(cudaMalloc((void **)&bn_counts, sizeof(vidType)*nwarps*max_depth));
  CUDA_SAFE_CALL(cudaMemset(bn_counts, 0, sizeof(vidType)*nwarps*max_depth));
  CUDA_SAFE_CALL(cudaMalloc((void **)&idx_counts, sizeof(vidType)*nwarps*max_depth));
  CUDA_SAFE_CALL(cudaMemset(idx_counts, 0, sizeof(vidType)*nwarps*max_depth));
  CUDA_SAFE_CALL(cudaMalloc((void **)&embeddings, sizeof(vidType)*nwarps*max_depth));
  CUDA_SAFE_CALL(cudaMemset(embeddings, 0, sizeof(vidType)*nwarps*max_depth));
  CUDA_SAFE_CALL(cudaMalloc((void **)&idx_embeddings, sizeof(vidType)*nwarps*max_depth));
  CUDA_SAFE_CALL(cudaMalloc((void **)&bns, sizeof(vidType)*nwarps*max_depth*max_depth));
  CUDA_SAFE_CALL(cudaMemset(bns, 0, sizeof(vidType)*nwarps*max_depth*max_depth));
  int nc = 0;
  for (int j = 1; j < max_depth; ++j) nc += candidates_count[order[j]];
  CUDA_SAFE_CALL(cudaMalloc((void **)&valid_candidates, sizeof(vidType)*nwarps*max_depth*nc));
  CUDA_SAFE_CALL(cudaMalloc((void **)&masks, sizeof(bool)*nwarps*nv));
  CUDA_SAFE_CALL(cudaMemset(masks, 0, sizeof(bool)*nwarps*nv));
  CUDA_SAFE_CALL(cudaMalloc((void **)&visited_query_vertices, sizeof(bool)*max_depth));
  CUDA_SAFE_CALL(cudaMemset(visited_query_vertices, 0, sizeof(bool)*max_depth));
  CUDA_SAFE_CALL(cudaMalloc((void **)&valid_candidate_idxes, sizeof(int)*nwarps*max_depth*max_candidates_num));
  CUDA_SAFE_CALL(cudaMalloc((void **)&temp_buffers, sizeof(int)*nwarps*max_candidates_num));

  GraphGPU g_qg(*qg);
  vidType* d_order;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_order, sizeof(vidType)*qg->size()));
  CUDA_SAFE_CALL(cudaMemcpy(d_order, &order, sizeof(vidType)*qg->size(), cudaMemcpyHostToDevice));
  generateBN<<<(nwarps+1)/128, 128>>>(nwarps, g_qg, d_order, bns, bn_counts);
  std::cout << "Start GPU GQL exploration, number of parallel tasks: " << num << "\n";

  Timer t;
  t.Start();
  warp_GQL<<<nblocks, nthreads>>>(gg, num, g_qg, idxes, bn_counts, idx_counts, embeddings, idx_embeddings, bns, masks, valid_candidate_idxes, temp_buffers, d_total);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  cudaProfilerStop();

  std::cout << "runtime [" << name << "] = " << t.Seconds() << " sec\n";
  CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_total));
  return h_total; 
}
/*
uint64_t QueryPlan::exploreLFTJGPU(const Graph *dg, const Pattern *qg, VertexLists &candidates, 
                                VertexList &candidates_count, VertexList &order,
                                Edges ***edge_matrix, size_t &call_count) {
  size_t memsize = print_device_info(0);
  auto nv = dg->num_vertices();
  auto ne = dg->num_edges();
  auto md = dg->get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";

  int max_depth = qg->size();
  auto start_vertex = order[0];
  auto max_candidates_num = candidates_count[0];
  for (int j = 1; j < max_depth; ++j) {
    auto cur_count = candidates_count[j];
    if (cur_count > max_candidates_num)
      max_candidates_num = cur_count;
  }
  uint64_t embedding_cnt = 0;
  auto num = candidates_count[start_vertex]; // number of candidates at level 0 (i.e. #v0 for u0)

  size_t nthreads = BLOCK_SIZE;
  size_t nwarps = BLK_SZ/WARP_SIZE;
  size_t nblocks = (num-1)/WARPS_PER_BLOCK+1;
  if (nblocks > 65536) nblocks = 65536;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(warp_LFTJ, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  //size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  //nblocks = std::min(max_blocks, nblocks);
  std::cout << "CUDA triangle counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  AccType h_total = 0, *d_total;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total, &h_total, sizeof(AccType), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  vidType* idxes, bn_counts, idx_counts, embeddings, idx_embeddings, bns;
  bool* masks;
  CUDA_SAFE_CALL(cudaMalloc((void **)&idxes, sizeof(vidType)*nwarps*max_depth));
  CUDA_SAFE_CALL(cudaMemset(idxes, 0, sizeof(vidType)*nwarps*max_depth));
  CUDA_SAFE_CALL(cudaMalloc((void **)&bn_counts, sizeof(vidType)*nwarps*max_depth));
  CUDA_SAFE_CALL(cudaMemset(bn_counts, 0, sizeof(vidType)*nwarps*max_depth));
  CUDA_SAFE_CALL(cudaMalloc((void **)&idx_counts, sizeof(vidType)*nwarps*max_depth));
  CUDA_SAFE_CALL(cudaMemset(idx_counts, 0, sizeof(vidType)*nwarps*max_depth));
  CUDA_SAFE_CALL(cudaMalloc((void **)&embeddings, sizeof(vidType)*nwarps*max_depth));
  CUDA_SAFE_CALL(cudaMemset(embeddings, 0, sizeof(vidType)*nwarps*max_depth));
  CUDA_SAFE_CALL(cudaMalloc((void **)&idx_embeddings, sizeof(vidTyoe)*nwarps*max_depth));
  CUDA_SAFE_CALL(cudaMalloc((void **)&bns, sizeof(vidType)*nwarps*max_depth*max_depth));
  CUDA_SAFE_CALL(cudaMemset(bns, 0, sizeof(vidType)*nwarps*max_depth*max_depth));
  CUDA_SAFE_CALL(cudaMalloc((void **)&masks, sizeof(bool)*nwarps*nv));
  CUDA_SAFE_CALL(cudaMemset(masks, 0, sizeof(bool)*nwarps*nv));
  CUDA_SAFE_CALL(cudaMalloc((void **)&valid_candidate_idxes, sizeof(int)*nwarps*max_depth*max_candidates_num));
  CUDA_SAFE_CALL(cudaMalloc((void **)&temp_buffers, sizeof(int)*nwarps*max_candidates_num));

  GraphGPU g_qg(*qg);
  vidType* d_order;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_order, sizeof(vidType)*qg->size()));
  CUDA_SAFE_CALL(cudaMemcpy(d_order, &order, sizeof(vidType)*qg->size(), cudaMemcpyHostToDevice));
  generateBN<<<(nwarps+1)/128, 128>>>(nwarps, g_qg, d_order, bns, bn_counts);
  std::cout << "Start parallel LFTJ exploration, number of parallel tasks: " << num << "\n";

  Timer t;
  t.Start();
  warp_LFTJ<<<nblocks, nthreads>>>(num, g_qg, idxes, bn_counts, idx_counts, embeddings, idx_embeddings, bns, masks, valid_candidate_idxes, temp_buffers, d_total);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  cudaProfilerStop();

  std::cout << "runtime [" << name << "] = " << t.Seconds() << " sec\n";
  CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_total));
  return h_total;
}
*/
