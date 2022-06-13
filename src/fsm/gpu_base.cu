// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>

#include "graph.h"
#include "cutil_subset.h"
#include <cub/cub.cuh>
#define USE_PID
#define USE_DOMAIN
#define EDGE_INDUCED
#define ENABLE_LABEL
#include "pangolin_gpu/miner.cuh"
#include "bitsets.h"
#include "fsm_operations.cuh"
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
typedef int flag_t;
#define MAX_NUM_PATTERNS 8000
//#define MAX_NUM_PATTERNS 26100

__global__ void extend_alloc(eidType m, eidType start_eid, int level, GraphGPU g, 
                             EmbeddingList emb_list, int *num_new_emb) {
  int tid = threadIdx.x;
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int vid[BLOCK_SIZE][MAX_FSM_PATTERN_SIZE];
  __shared__ BYTE his[BLOCK_SIZE][MAX_FSM_PATTERN_SIZE];
  if (pos < m) {
    emb_list.get_edge_embedding(level, start_eid+pos, vid[tid], his[tid]);
    num_new_emb[pos] = 0;
    for (int i = 0; i < level+1; ++i) {
      auto src = vid[tid][i];
      auto row_begin = g.edge_begin(src);
      auto row_end = g.edge_end(src);
      for (auto e = row_begin; e < row_end; e++) {
        auto dst = g.getEdgeDst(e);
        if (!is_edge_automorphism(level+1, vid[tid], his[tid], i, src, dst))
          num_new_emb[pos] ++;
      }
    }
  }
}

__global__ void extend_insert(eidType m, eidType start_eid, int level,
                              GraphGPU graph, EmbeddingList emb_list, int *indices) {
  int tid = threadIdx.x;
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int vids[BLOCK_SIZE][MAX_FSM_PATTERN_SIZE];
  __shared__ BYTE his[BLOCK_SIZE][MAX_FSM_PATTERN_SIZE];
  if (pos < m) {
    emb_list.get_edge_embedding(level, start_eid+pos, vids[tid], his[tid]);
    auto start = indices[pos];
    for (int i = 0; i < level+1; ++i) {
      auto src = vids[tid][i];
      auto row_begin = graph.edge_begin(src);
      auto row_end = graph.edge_end(src);
      for (auto e = row_begin; e < row_end; e++) {
        auto dst = graph.getEdgeDst(e);
        if (!is_edge_automorphism(level+1, vids[tid], his[tid], i, src, dst)) {
          emb_list.set_idx(level+1, start, start_eid+pos);
          emb_list.set_his(level+1, start, i);
          emb_list.set_vid(level+1, start++, dst);
        }
      }
    }
  }
}

__global__ void init_aggregate(eidType nedges, GraphGPU g, int nlabels, int threshold, int *set_id,
                               int *pids, Bitsets small_sets, Bitsets large_sets) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < nedges) {
    auto src = g.get_src(pos);
    auto dst = g.get_dst(pos);
    if (g.is_freq_vertex(src, threshold) && g.is_freq_vertex(dst, threshold)) {
      auto src_label = g.getData(src);
      auto dst_label = g.getData(dst);
      int pid = 0;
      if (src_label <= dst_label)
        pid = get_init_pattern_id(src_label, dst_label, nlabels);
      else pid = get_init_pattern_id(dst_label, src_label, nlabels);
      pids[pos] = pid;
      pid = set_id[pid];
      if (src_label < dst_label) {
        small_sets.set(pid, src);
        large_sets.set(pid, dst);
      } else if (src_label > dst_label) {
        small_sets.set(pid, dst);
        large_sets.set(pid, src);
      } else {
        small_sets.set(pid, src);
        small_sets.set(pid, dst);
        large_sets.set(pid, src);
        large_sets.set(pid, dst);
      }
    }
  }
}

__global__ void count_ones(int id, Bitsets sets, int *count) {
  typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduce;
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int num = 0;
  if(pos < sets.vec_size())
    num = sets.count_num_ones(id, pos);
  int block_total = BlockReduce(temp_storage).Sum(num);
  if(threadIdx.x == 0) atomicAdd(count, block_total);
}

int init_support_count(Graph &g, int nlabels, int threshold, 
                       Bitsets small_sets, Bitsets large_sets, 
                       int *set_id, mask_t *init_support_map) {
  int num_freq_patterns = 0;
  int m = g.size();
  for (int i = 0; i < nlabels+1; i++) {
    if (g.getLabelsFrequency(i) < threshold) continue;
    for (int j = 0; j < nlabels+1; j++) {
      if (g.getLabelsFrequency(j) < threshold) continue;
      int a, b, *d_count;
      auto pid = get_init_pattern_id(i, j, nlabels);
      auto sid = set_id[pid];
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_count, sizeof(int)));
      CUDA_SAFE_CALL(cudaMemset(d_count, 0, sizeof(int)));
      count_ones<<<(m-1)/256+1, 256>>>(sid, small_sets, d_count);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      CUDA_SAFE_CALL(cudaMemcpy(&a, d_count, sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemset(d_count, 0, sizeof(int)));
      count_ones<<<(m-1)/256+1, 256>>>(sid, large_sets, d_count);
      CUDA_SAFE_CALL(cudaMemcpy(&b, d_count, sizeof(int), cudaMemcpyDeviceToHost));
      auto support = a < b ? a : b;
      if (support >= threshold) {
        init_support_map[pid] = 1;
        num_freq_patterns ++;
      } 
    }
  }
  return num_freq_patterns;
}

int support_count(Graph &g, int nlabels, int threshold, int *set_id, 
                  Bitsets small_sets, Bitsets middle_sets, Bitsets large_sets, mask_t *support_map) {
  auto m = g.size();
  int num_freq_patterns = 0;
  for (int i = 0; i < nlabels+1; i++) {
    if (g.getLabelsFrequency(i) < threshold) continue;
    for (int j = 0; j < nlabels+1; j++) {
      if (g.getLabelsFrequency(j) < threshold) continue;
      for (int l = 0; l <= j; l++) {
        if (g.getLabelsFrequency(l) < threshold) continue;
        int a, b, c, *d_count;
        auto pid = get_pattern_id(i, j, l, nlabels);
        auto sid = set_id[pid];
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_count, sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset(d_count, 0, sizeof(int)));
        count_ones<<<(m-1)/256+1, 256>>>(sid, small_sets, d_count);
        CUDA_SAFE_CALL(cudaMemcpy(&a, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemset(d_count, 0, sizeof(int)));
        count_ones<<<(m-1)/256+1, 256>>>(sid, large_sets, d_count);
        CUDA_SAFE_CALL(cudaMemcpy(&b, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemset(d_count, 0, sizeof(int)));
        count_ones<<<(m-1)/256+1, 256>>>(sid, middle_sets, d_count);
        CUDA_SAFE_CALL(cudaMemcpy(&c, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        auto small = a < b ? a : b;
        auto support = small < c ? small : c;
        if (support >= threshold) {
          support_map[pid] = 1;
          num_freq_patterns ++;
        }
      }
    }
  }
  return num_freq_patterns;
}

__global__ void init_filter_check(int m, int *pids, mask_t *init_support_map, flag_t* is_frequent_emb) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if(pos < m) {
    auto pid = pids[pos];
    auto is_frequent = init_support_map[pid];
    if (is_frequent) is_frequent_emb[pos] = 1;
    else is_frequent_emb[pos] = 0;
  }
}

__global__ void init_filter(eidType m, GraphGPU g, int *indices, 
                            flag_t* is_frequent_emb, EmbeddingList emb_list) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < m) {
    if (is_frequent_emb[pos]) {
      auto src = g.get_src(pos);
      auto dst = g.get_dst(pos);
      auto start = indices[pos];
      emb_list.set_vid(1, start, dst);
      emb_list.set_idx(1, start, src);
    }
  }
}

__global__ void compute_pattern_id(eidType num_emb, int level, GraphGPU graph, 
                                   EmbeddingList emb_list, int nlabels, 
                                   int *pids, int *ne) {
  int tid = threadIdx.x;
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int vids[BLOCK_SIZE][MAX_FSM_PATTERN_SIZE];
  __shared__ BYTE his[BLOCK_SIZE][MAX_FSM_PATTERN_SIZE];
  if (pos < num_emb) {
    emb_list.get_edge_embedding(level, pos, vids[tid], his[tid]);
    auto n = level+1;
    assert(n < 4);
    auto first = vids[tid][0];
    auto second = vids[tid][1];
    auto third = vids[tid][2];
    BYTE l0 = graph.getData(first);
    BYTE l1 = graph.getData(second);
    BYTE l2 = graph.getData(third);
    BYTE h2 = his[tid][2];
    int pid = 0;
    if (n == 3) {
      if (h2 == 0) {
        if (l1 < l2) {
          pid = get_pattern_id(l0, l2, l1, nlabels);
        } else {
          pid = get_pattern_id(l0, l1, l2, nlabels);
        }
      } else {
        assert(h2 == 1);
        if (l0 < l2) {
          pid = get_pattern_id(l1, l2, l0, nlabels);
        } else {
          pid = get_pattern_id(l1, l0, l2, nlabels);
        }
      }
    } else {
    }
    pids[pos] = pid;
    //atomicAdd(&ne[pid], 1);
  }
}

__global__ void find_candidate_patterns(int num_patterns, int *ne, int minsup, int *id_map, int *num_new_patterns) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < num_patterns) {
    if (ne[pos] >= minsup) {
      auto new_id = atomicAdd(num_new_patterns, 1);
      id_map[pos] = new_id;
    }
  }
}

__global__ void aggregate(int num_emb, int level, GraphGPU graph, EmbeddingList emb_list, 
                          int *pids, int *ne, int *id_map, int threshold, 
                          Bitsets small_sets, Bitsets middle_sets, Bitsets large_sets) {
  int tid = threadIdx.x;
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int vids[BLOCK_SIZE][MAX_FSM_PATTERN_SIZE];
  __shared__ BYTE his[BLOCK_SIZE][MAX_FSM_PATTERN_SIZE];
  auto nv = graph.size();
  if (pos < num_emb) {
    emb_list.get_edge_embedding(level, pos, vids[tid], his[tid]);
    auto n = level+1;
    assert(n == 3);
    auto first = vids[tid][0];
    auto second = vids[tid][1];
    auto third = vids[tid][2];
    BYTE l0 = graph.getData(first);
    BYTE l1 = graph.getData(second);
    BYTE l2 = graph.getData(third);
    BYTE h2 = his[tid][2];
    int small, middle, large;
    auto pid = pids[pos];
    //if (ne[pid] >= threshold) {
      pid = id_map[pid];
    if (pid != -1) {
      if (h2 == 0) {
        middle = first;
        if (l1 < l2) {
          small = second;
          large = third;
        } else {
          small = third;
          large = second;
        }
        small_sets.set(pid, small);
        middle_sets.set(pid, middle);
        large_sets.set(pid, large);
        if (l1 == l2) {
          small_sets.set(pid, large);
          large_sets.set(pid, small);
        }
      } else {
        assert(h2 == 1);
        middle = second;
        if (l0 < l2) {
          small = first;
          large = third;
        } else {
          small = third;
          large = first;
        }
        small_sets.set(pid, small);
        middle_sets.set(pid, middle);
        large_sets.set(pid, large);
        if (l0 == l2) {
          small_sets.set(pid, large);
          large_sets.set(pid, small);
        }
      }
    }
  }
}

void parallel_prefix_sum(int n, int *in, int *out) {
  int total = 0;
  for (size_t i = 0; i < n; i++) {
    out[i] = total;
    total += in[i];
  }
  out[n] = total;
}

void FsmSolver(Graph &g, int k, int minsup, int &total_num) {
  assert(k >= 2);
  size_t memsize = print_device_info(0);
  auto nv = g.num_vertices();
  auto ne = g.num_edges();
  auto md = g.get_max_degree();
  auto mv = g.get_max_label_frequency();
  int nlabels = g.get_vertex_classes();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize/1024/1024/1024
            << " GB, graph_mem = " << mem_graph/1024/1024/1024 << " GB\n";
  if (memsize < mem_graph) std::cout << "Graph too large. Unified Memory (UM) required\n";

  GraphGPU gg(g);
  auto nnz = gg.init_edgelist(g, 1, 1);
  auto n_freq_labels = g.get_frequent_labels(minsup);
  std::cout << "Number of frequent labels: " << n_freq_labels << std::endl;
  int num_init_freq_patterns = (n_freq_labels+1)*(n_freq_labels+1);
  int num_init_patterns = (nlabels+1)*(nlabels+1);
  std::cout << "Number of init patterns: " << num_init_patterns << std::endl;
  std::cout << "Number of init frequent patterns: " << num_init_freq_patterns << std::endl;
  std::cout << "number of single-edge embeddings: " << nnz << "\n";
  int *pids;
  CUDA_SAFE_CALL(cudaMalloc((void **)&pids, sizeof(int)*nnz));
  std::vector<mask_t> h_init_support_map(num_init_patterns, 0);
  mask_t *d_init_support_map;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_init_support_map, sizeof(mask_t)*num_init_patterns));

  auto mlf = g.get_max_label_frequency();
  std::cout << "Allocating vertex sets\n";
  Bitsets small_sets, large_sets, middle_sets;
  small_sets.alloc(MAX_NUM_PATTERNS, nv);
  large_sets.alloc(MAX_NUM_PATTERNS, nv);
  middle_sets.alloc(MAX_NUM_PATTERNS, nv);
  small_sets.set_size(num_init_freq_patterns, nv);
  large_sets.set_size(num_init_freq_patterns, nv);

  std::cout << "Computing mapping from pattern id to set id\n";
  std::vector<int> set_id(num_init_patterns, -1);
  int *d_set_id;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_set_id, sizeof(int)*num_init_patterns));
  int si = 0;
  for (int i = 0; i < nlabels+1; i++) {
    if (g.getLabelsFrequency(i) < minsup) continue;
    for (int j = 0; j < nlabels+1; j++) {
      if (g.getLabelsFrequency(j) < minsup) continue;
      int pid = get_init_pattern_id(i, j, nlabels);
      set_id[pid] = si++;
    }
  }
  CUDA_SAFE_CALL(cudaMemcpy(d_set_id, &set_id[0], sizeof(int) * (nlabels+1)*(nlabels+1), cudaMemcpyHostToDevice));
  std::cout << "Number of edge patterns: " << num_init_patterns << ", possible frequent edge patterns: " << si << "\n";

  int *emb_count_per_pattern, *d_id_map;
  //int num_wedge_patterns = (nlabels+1)*((nlabels+1)*(nlabels)/2+(nlabels+1));
  int num_wedge_patterns = (nlabels+1)*(nlabels+1)*(nlabels+1);
  std::vector<mask_t> support_map(num_wedge_patterns, 0);
  std::vector<int> id_map(num_wedge_patterns, -1);
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_id_map, sizeof(int)*num_wedge_patterns));
  int num_freq_wedge_patterns = 0;
  for (int i = 0; i < nlabels+1; i++) {
    if (g.getLabelsFrequency(i) < minsup) continue;
    for (int j = 0; j < nlabels+1; j++) {
      if (g.getLabelsFrequency(j) < minsup) continue;
      for (int l = 0; l <= j; l++) {
        if (g.getLabelsFrequency(l) < minsup) continue;
        int pid = get_pattern_id(i, j, l, nlabels);
        id_map[pid] = num_freq_wedge_patterns++;
      }
    }
  }
  CUDA_SAFE_CALL(cudaMemcpy(d_id_map, &id_map[0], sizeof(int) * num_wedge_patterns, cudaMemcpyHostToDevice));
  std::cout << "Number of wedge patterns: " << num_wedge_patterns << ", possible frequent wedge patterns: " << num_freq_wedge_patterns << "\n";
 
  int nthreads = BLOCK_SIZE;
  int nblocks = (nnz-1)/nthreads+1;
  int *d_num_new_patterns;
  int h_num_new_patterns = 0;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_num_new_patterns, sizeof(int)));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  std::cout << "CUDA FSM (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  Timer t;
  t.Start();
  int level = 1;
  init_aggregate<<<nblocks, nthreads>>>(nnz, gg, nlabels, minsup, d_set_id, pids, small_sets, large_sets);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  int num_freq_patterns = init_support_count(g, nlabels, minsup, small_sets, large_sets, &set_id[0], h_init_support_map.data());
  small_sets.clear();
  large_sets.clear();
  total_num += num_freq_patterns;
  if (num_freq_patterns == 0) {
    std::cout << "No frequent pattern found\n\n";
    return;
  }
  std::cout << "Number of frequent single-edge patterns: " << num_freq_patterns << "\n";

  CUDA_SAFE_CALL(cudaMemcpy(d_init_support_map, &h_init_support_map[0], sizeof(mask_t) * num_init_patterns, cudaMemcpyHostToDevice));
  flag_t* is_frequent_emb;
  CUDA_SAFE_CALL(cudaMalloc((void **)&is_frequent_emb, sizeof(flag_t)*(nnz+1)));
  //CUDA_SAFE_CALL(cudaMemset(is_frequent_emb, 0, sizeof(flag_t)*(nnz+1)));
  init_filter_check<<<nblocks, nthreads>>>(nnz, pids, d_init_support_map, is_frequent_emb);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaFree(pids));
  CUDA_SAFE_CALL(cudaFree(d_init_support_map));

  int *num_new_emb, *indices;
  CUDA_SAFE_CALL(cudaMalloc((void **)&indices, sizeof(int) * (nnz+1)));
  thrust::exclusive_scan(thrust::device, is_frequent_emb, is_frequent_emb+nnz+1, indices);
  int new_size = 0;
  CUDA_SAFE_CALL(cudaMemcpy(&new_size, &indices[nnz], sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "Number of embeddings after pruning: " << new_size << "\n";
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  EmbeddingList emb_list(k+1);
  emb_list.add_level(new_size);
  init_filter<<<nblocks, nthreads>>>(nnz, gg, indices, is_frequent_emb, emb_list);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaFree(is_frequent_emb));
  CUDA_SAFE_CALL(cudaFree(indices));
  gg.clean_edgelist();
  nnz = emb_list.size();

  while (1) {
    std::cout << "number of embeddings in level " << level << ": " << nnz << "\n";
    //int num_patterns = nlabels*num_init_patterns;
    //std::cout << "Number of patterns in level " << level+1 << ": " << num_patterns << std::endl;
    small_sets.set_size(num_freq_wedge_patterns, nv);
    large_sets.set_size(num_freq_wedge_patterns, nv);
    middle_sets.set_size(num_freq_wedge_patterns, nv);
    std::cout << "Resizing sets to " << num_freq_wedge_patterns << std::endl;

    auto emb_block_size = 640*128;
    if (nnz < emb_block_size) 
      emb_block_size = nnz;
    CUDA_SAFE_CALL(cudaMalloc((void **)&num_new_emb, sizeof(int) * (emb_block_size+1)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&indices, sizeof(int) * (emb_block_size+1)));
    size_t total_num_new_emb = 0;
    for (eidType eid = 0; eid < nnz; eid += emb_block_size) {
      if (nnz - eid < emb_block_size) emb_block_size = nnz - eid;
      nblocks = (emb_block_size-1)/nthreads+1;
      //CUDA_SAFE_CALL(cudaMemset(num_new_emb, 0, sizeof(int)*(emb_block_size+1)));
      //std::cout << "extending embeddings from " << eid << " to " << eid+emb_block_size << "\n";
      extend_alloc<<<nblocks, nthreads>>>(emb_block_size, eid, level, gg, emb_list, num_new_emb);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      thrust::exclusive_scan(thrust::device, num_new_emb, num_new_emb+emb_block_size+1, indices);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      CUDA_SAFE_CALL(cudaMemcpy(&new_size, &indices[emb_block_size], sizeof(int), cudaMemcpyDeviceToHost));
      //std::cout << "number of new embeddings: " << new_size << "\n";
      if (eid==0) emb_list.add_level();
      emb_list.resize_last_level(new_size);
      extend_insert<<<nblocks, nthreads>>>(emb_block_size, eid, level, gg, emb_list, indices);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      total_num_new_emb += new_size;

      nblocks = (new_size-1)/nthreads+1;
      //CUDA_SAFE_CALL(cudaMalloc((void **)&emb_count_per_pattern, sizeof(int)*num_patterns));
      //CUDA_SAFE_CALL(cudaMalloc((void **)&d_id_map, sizeof(int)*num_patterns));
      CUDA_SAFE_CALL(cudaMalloc((void **)&pids, sizeof(int)*new_size));
      //std::cout << "compute pattern id for the new embeddings\n";
      compute_pattern_id<<<nblocks, nthreads>>>(new_size, level+1, gg, emb_list, nlabels, pids, emb_count_per_pattern);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      //CUDA_SAFE_CALL(cudaMemset(d_num_new_patterns, 0, sizeof(int)));
      //find_candidate_patterns<<<(num_patterns-1)/nthreads+1, nthreads>>>(num_patterns, emb_count_per_pattern, minsup, id_map, d_num_new_patterns);
      //CUDA_SAFE_CALL(cudaDeviceSynchronize());
      //CUDA_SAFE_CALL(cudaMemcpy(&h_num_new_patterns, d_num_new_patterns, sizeof(int), cudaMemcpyDeviceToHost));
      //std::cout << "Number of candidate patterns in level " << level+1 << ": " << h_num_new_patterns << std::endl;

      //std::cout << "aggregate new embeddings\n";
      aggregate<<<nblocks, nthreads>>>(new_size, level+1, gg, emb_list, pids, emb_count_per_pattern, d_id_map, minsup, small_sets, middle_sets, large_sets);
      //CUDA_SAFE_CALL(cudaFree(emb_count_per_pattern));
      //CUDA_SAFE_CALL(cudaFree(id_map));
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      CUDA_SAFE_CALL(cudaFree(pids));
    }
    level ++;
    CUDA_SAFE_CALL(cudaFree(num_new_emb));
    CUDA_SAFE_CALL(cudaFree(indices));
    nnz = total_num_new_emb;
    std::cout << "number of new embeddings: " << nnz << "\n";
    num_freq_patterns = support_count(g, nlabels, minsup, &id_map[0], small_sets, middle_sets, large_sets, &support_map[0]);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    std::cout << "num_frequent_patterns: " << num_freq_patterns << "\n";
    total_num += num_freq_patterns;
    if (num_freq_patterns == 0) break;
    if (level == k) break;
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  std::cout << "runtime [gpu_base] = " << t.Seconds() << " sec\n";
}

