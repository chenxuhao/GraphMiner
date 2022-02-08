#include "scheduler.h"

inline int64_t Scheduler::hop2_workload(Graph &g, vidType src, vidType dst) {
  int64_t sum = 0;
  for (auto v : g.N(src)) {
    sum += g.get_degree(v);
  }
  for (auto v : g.N(dst)) {
    sum += g.get_degree(v);
  }
  return sum;
}

inline int64_t Scheduler::workload_estimate(Graph &g, vidType src, vidType dst) {
  //return hop2_workload(g, src, dst); // 2-hop
  //return g.get_degree(src) + g.get_degree(dst); // 1-hop
  return std::min(g.get_degree(src), g.get_degree(dst)); // minimum of two for cliques
  //return g.get_degree(src); 
  //return 1;
}

inline int Scheduler::smallest_score_id(int n, int64_t* scores) {
  int id = 0;
  auto min_score = scores[0];
  for (int i = 1; i < n; i++) {
    if (scores[i] < min_score) {
      min_score = scores[i];
      id = i;
    }
  }
  return id;
}

std::vector<eidType> Scheduler::round_robin(int n, Graph &g, std::vector<vidType*> &src_ptrs, std::vector<vidType*> &dst_ptrs, int stride) {
  auto nnz = g.get_num_tasks();
  assert(nnz > 8192); // if edgelist is too small, no need to split
  std::cout << "split edgelist with chunk size of " << stride << " using chunked round robin\n";
  src_ptrs.resize(n);
  dst_ptrs.resize(n);
 
  eidType total_num_chunks = (nnz-1) / stride + 1;
  eidType nchunks_per_queue = total_num_chunks / n;
  std::vector<eidType> lens(n, stride*nchunks_per_queue);
  if (total_num_chunks % n != 0) {
    for (int i = 0; i < n; i++) {
      if (i+1 == int(total_num_chunks % n)) {
        lens[i] += nnz%stride == 0 ? stride : nnz%stride;
      } else if (i+1 < int(total_num_chunks % n)) {
        lens[i] += stride;
      }
    }
  } else {
    lens[n-1] = lens[n-1] + nnz%stride - stride;
  }
  for (int i = 0; i < n; i++) {
    src_ptrs[i] = new vidType[lens[i]];
    dst_ptrs[i] = new vidType[lens[i]];
  }
  auto src_list = g.get_src_ptr();
  auto dst_list = g.get_dst_ptr();
  #pragma omp parallel for
  for (eidType chunk_id = 0; chunk_id < nchunks_per_queue; chunk_id++) {
    eidType begin = chunk_id * n * stride;
    for (int qid = 0; qid < n; qid++) {
      eidType pos = begin + qid*stride;
      int size = stride;
      if ((total_num_chunks % n == 0) && (chunk_id == nchunks_per_queue-1) && (qid == n-1))
        size = nnz%stride;
      std::copy(src_list+pos, src_list+pos+size, src_ptrs[qid]+chunk_id*stride);
      std::copy(dst_list+pos, dst_list+pos+size, dst_ptrs[qid]+chunk_id*stride);
    }
  }
  eidType begin = nchunks_per_queue*n*stride;
  for (int i = 0; i < n; i++) {
    eidType pos = begin + i*stride;
    if (i+1 == int(total_num_chunks % n)) {
      std::copy(src_list+pos, src_list+nnz, src_ptrs[i]+nchunks_per_queue*stride);
      std::copy(dst_list+pos, dst_list+nnz, dst_ptrs[i]+nchunks_per_queue*stride);
    } else if (i+1 < int(total_num_chunks % n)) {
      std::copy(src_list+pos, src_list+pos+stride, src_ptrs[i]+nchunks_per_queue*stride);
      std::copy(dst_list+pos, dst_list+pos+stride, dst_ptrs[i]+nchunks_per_queue*stride);
    }
  }
  return lens;
}

std::vector<vidType> construct_index(vidType nv, eidType nnz, vidType *vertices) {
  std::vector<vidType> sizes(nv, 0);
  for (eidType id = 0; id < nnz; id++) {
    auto v = vertices[id];
    sizes[v] ++;
  }
  return sizes;
}

#pragma omp declare reduction(vec64_plus : std::vector<eidType> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<eidType>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

std::vector<eidType> Scheduler::vertex_chunking(int n, Graph &g, std::vector<vidType*> &src_ptrs, std::vector<vidType*> &dst_ptrs, int stride) {
  auto nnz = g.get_num_tasks();
  auto src_list = g.get_src_ptr();
  auto dst_list = g.get_dst_ptr();
  auto nv = g.V();
  //auto sizes = construct_index(nv, nnz, src_list);
  const auto &sizes = g.get_sizes();
  src_ptrs.resize(n);
  dst_ptrs.resize(n);
  std::vector<eidType> lens(n, 0);
  #pragma omp parallel for reduction(vec64_plus:lens)
  for (vidType v = 0; v < nv; v++) {
    int i = (v/stride)%n;
    lens[i] += sizes[v];
  }
  for (int i = 0; i < n; i++) {
    src_ptrs[i] = new vidType[lens[i]];
    dst_ptrs[i] = new vidType[lens[i]];
  }

  std::vector<int> idx(n, 0);
  for (eidType eid = 0; eid < nnz; eid++) {
    auto v = src_list[eid];
    auto u = dst_list[eid];
    int qid = (v/stride)%n;
    src_ptrs[qid][idx[qid]] = v;
    dst_ptrs[qid][idx[qid]] = u;
    idx[qid] ++;
  }
  return lens;
}

// split the edgelist into n subsets; each subset has a src list and a dst list; strid is the chunk size
std::vector<eidType> Scheduler::least_first(int n, Graph &g, std::vector<vidType*> &src_ptrs, std::vector<vidType*> &dst_ptrs, int stride) {
  auto nnz = g.get_num_tasks();
  assert(nnz > 8192); // if edgelist is too small, no need to split
  std::cout << "split edgelist with chunk size of " << stride << " using chunked least first\n";
  srcs.resize(n);
  dsts.resize(n);
  int64_t* scores = new int64_t[n];
  std::fill(scores, scores+n, 0);
  std::vector<int> num_chunks(n, 1);
  std::vector<eidType> lens(n);

  eidType init_stride = stride;
  eidType pos = 0;
  for (int i = 0; i < n; i++) {
    srcs[i].resize(init_stride);
    dsts[i].resize(init_stride);
  }
  auto src_list = g.get_src_ptr();
  auto dst_list = g.get_dst_ptr();
  //std::cout << "assign the first chunk, size = " << init_stride << "\n";
  for (int i = 0; i < n; i++) {
    //#pragma omp parallel for reduction(+:scores[i])
    for (eidType j = i*init_stride; j < (i+1)*init_stride; j++) {
      assert(j < nnz);
      auto src = src_list[j];
      auto dst = dst_list[j];
      scores[i] += workload_estimate(g, src, dst);
    }
    lens[i] += init_stride;
  }
  for (int i = 0; i < n; i++) {
    std::copy(src_list+pos, src_list+pos+init_stride, srcs[i].begin());
    std::copy(dst_list+pos, dst_list+pos+init_stride, dsts[i].begin());
    pos += init_stride;
  }
  assert(pos < nnz);
  auto id = smallest_score_id(n, scores);
  //std::cout << "assign one chunk a time\n";
  while (pos + stride < nnz) {
    //#pragma omp parallel for reduction(+:scores[id])
    for (int j = 0; j < stride; j++) {
      auto src = src_list[pos+j];
      auto dst = dst_list[pos+j];
      scores[id] += workload_estimate(g, src, dst);
    }
    lens[id] += stride;
    num_chunks[id] ++;
    auto curr_size = srcs[id].size();
    srcs[id].resize(curr_size+stride);
    dsts[id].resize(curr_size+stride);
    std::copy(src_list+pos, src_list+pos+stride, &srcs[id][curr_size]);
    std::copy(dst_list+pos, dst_list+pos+stride, &dsts[id][curr_size]);
    pos += stride;
    id = smallest_score_id(n, scores);
  }
  //std::cout << "assign the last chunk\n";
  if (pos < nnz) {
    num_chunks[id] ++;
    lens[id] += nnz-pos;
  }
  while (pos < nnz) {
    srcs[id].push_back(src_list[pos]);
    dsts[id].push_back(dst_list[pos]);
    pos++;
  }
  eidType total_len = 0;
  src_ptrs.resize(n);
  dst_ptrs.resize(n);
  //std::cout << "pass results\n";
  for (int i = 0; i < n; i++) {
    src_ptrs[i] = srcs[i].data();
    dst_ptrs[i] = dsts[i].data();
    assert(srcs[i].size() == dsts[i].size());
    //lens[i] = srcs[i].size();
    total_len += lens[i];
    std::cout << "partition " << i << " edgelist size = " << lens[i] << "\n";
  }
  for (int i = 0; i < n; i++)
    std::cout << "partition " << i << " assigned num_chunks = " << num_chunks[i] << "\n";
  assert(total_len == nnz);
  return lens;
}

