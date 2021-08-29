#include "graph.h"

void Graph::orientation() {
  std::cout << "Orientation enabled, using DAG\n";
  Timer t;
  t.Start();
  std::vector<vidType> degrees(n_vertices, 0);
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    degrees[v] = get_degree(v);
  }
  std::vector<vidType> new_degrees(n_vertices, 0);
  #pragma omp parallel for
  for (vidType src = 0; src < n_vertices; src ++) {
    for (auto dst : N(src)) {
      if (degrees[dst] > degrees[src] ||
          (degrees[dst] == degrees[src] && dst > src)) {
        new_degrees[src]++;
      }
    }
  }
  max_degree = *(std::max_element(new_degrees.begin(), new_degrees.end()));
  eidType *old_vertices = vertices;
  vidType *old_edges = edges;
  eidType *new_vertices = custom_alloc_global<eidType>(n_vertices+1);
  //prefix_sum<vidType,eidType>(new_degrees, new_vertices);
  parallel_prefix_sum<vidType,eidType>(new_degrees, new_vertices);
  auto num_edges = new_vertices[n_vertices];
  vidType *new_edges = custom_alloc_global<vidType>(num_edges);
  #pragma omp parallel for
  for (vidType src = 0; src < n_vertices; src ++) {
    auto begin = new_vertices[src];
    eidType offset = 0;
    for (auto dst : N(src)) {
      if (degrees[dst] > degrees[src] ||
          (degrees[dst] == degrees[src] && dst > src)) {
        new_edges[begin+offset] = dst;
        offset ++;
      }
    }
  }
  vertices = new_vertices;
  edges = new_edges;
  custom_free<eidType>(old_vertices, n_vertices);
  custom_free<vidType>(old_edges, n_edges);
  n_edges = num_edges;
  t.Stop();
  std::cout << "Time on generating the DAG: " << t.Seconds() << " sec\n";
}

void Graph::print_graph() {
  std::cout << "Printing the graph: \n";
  for (vidType n = 0; n < n_vertices; n++) {
    std::cout << "vertex " << n << ": degree = " 
      << get_degree(n) << " edgelist = [ ";
    for (auto e = edge_begin(n); e != edge_end(n); e++)
      std::cout << getEdgeDst(e) << " ";
    std::cout << "]\n";
  }
}

size_t Graph::init_edgelist(bool sym_break, bool ascend) {
  Timer t;
  t.Start();
  if (nnz != 0) return nnz; // already initialized
  nnz = E();
  if (sym_break) nnz = nnz/2;
  src_list = new vidType[nnz];
  if (sym_break) dst_list = new vidType[nnz];
  else dst_list = edges;
  size_t i = 0;
  for (vidType v = 0; v < V(); v ++) {
    for (auto u : N(v)) {
      if (u == v) continue; // no selfloops
      if (ascend) {
        if (sym_break && v > u) continue;  
      } else {
        if (sym_break && v < u) break;  
      }
      src_list[i] = v;
      if (sym_break) dst_list[i] = u;
      i ++;
    }
  }
  assert(i == nnz);
  t.Stop();
  std::cout << "Time on generating the edgelist: " << t.Seconds() << " sec\n";
  return nnz;
}

inline int Graph::smallest_score_id(size_t n, int64_t* scores) {
  int id = 0;
  auto min_score = scores[0];
  for (size_t i = 1; i < n; i++) {
    if (scores[i] < min_score) {
      min_score = scores[i];
      id = i;
    }
  }
  return id;
}

// split tasks into n subsets
std::vector<eidType> Graph::split_edgelist(int n, std::vector<vidType*> &src_ptrs, std::vector<vidType*> &dst_ptrs, int stride) {
  assert(nnz > 8192); // if edgelist is too small, no need to split
  //std::cout << "split edgelist\n";
  Timer t;
  t.Start();
  srcs.resize(n);
  dsts.resize(n);
  int64_t* scores = new int64_t[n];
  std::fill(scores, scores+n, 0);
  //size_t init_stride = nnz / 4 / n;
  size_t init_stride = stride;
  size_t pos = 0;
  for (int i = 0; i < n; i++) {
    srcs[i].resize(init_stride);
    dsts[i].resize(init_stride);
  }
  //std::cout << "assign the first chunk, size = " << init_stride << "\n";
  for (int i = 0; i < n; i++) {
    #pragma omp parallel for reduction(+:scores[i])
    for (size_t j = i*init_stride; j < (i+1)*init_stride; j++) {
      assert(j < nnz);
      auto src = src_list[j];
      auto dst = dst_list[j];
      scores[i] += get_degree(src) + get_degree(dst);
    }
  }
  for (int i = 0; i < n; i++) {
      //srcs[i].push_back(src);
      //dsts[i].push_back(dst);
      std::copy(src_list+pos, src_list+pos+init_stride, srcs[i].begin());
      std::copy(dst_list+pos, dst_list+pos+init_stride, dsts[i].begin());
    pos += init_stride;
  }
  assert(pos < nnz);
  auto id = smallest_score_id(n, scores);
  //std::cout << "assign one chunk a time\n";
  while (pos + stride < nnz) {
    #pragma omp parallel for reduction(+:scores[id])
    for (int j = 0; j < stride; j++) {
      auto src = src_list[pos+j];
      auto dst = dst_list[pos+j];
      //srcs[id].push_back(src);
      //dsts[id].push_back(dst);
      scores[id] += get_degree(src) + get_degree(dst);
    }
    auto curr_size = srcs[id].size();
    srcs[id].resize(curr_size+stride);
    dsts[id].resize(curr_size+stride);
    std::copy(src_list+pos, src_list+pos+stride, &srcs[id][curr_size]);
    std::copy(dst_list+pos, dst_list+pos+stride, &dsts[id][curr_size]);
    pos += stride;
    id = smallest_score_id(n, scores);
  }
  //std::cout << "assign the last chunk\n";
  while (pos < nnz) {
    srcs[id].push_back(src_list[pos]);
    dsts[id].push_back(dst_list[pos]);
    pos++;
  }
  std::vector<eidType> lens(n);
  size_t total_len = 0;
  src_ptrs.resize(n);
  dst_ptrs.resize(n);
  //std::cout << "pass results\n";
  for (int i = 0; i < n; i++) {
    src_ptrs[i] = srcs[i].data();
    dst_ptrs[i] = dsts[i].data();
    assert(srcs[i].size() == dsts[i].size());
    lens[i] = srcs[i].size();
    total_len += lens[i];
    std::cout << "partition " << i << " edgelist size = " << lens[i] << "\n";
  }
  assert(total_len == nnz);
  t.Stop();
  std::cout << "Time on splitting edgelist to GPUs: " << t.Seconds() << " sec\n";
  return lens;
}

bool Graph::is_connected(vidType v, vidType u) {
  auto v_deg = get_degree(v);
  auto u_deg = get_degree(u);
  bool found;
  if (v_deg < u_deg) {
    found = binary_search(u, edge_begin(v), edge_end(v));
  } else {
    found = binary_search(v, edge_begin(u), edge_end(u));
  }
  return found;
}

bool Graph::is_connected(std::vector<vidType> sg) {
  return false;
}

bool Graph::binary_search(vidType key, eidType begin, eidType end) {
  auto l = begin;
  auto r = end-1;
  while (r >= l) { 
    auto mid = l + (r - l) / 2;
    auto value = getEdgeDst(mid);
    if (value == key) return true;
    if (value < key) l = mid + 1; 
    else r = mid - 1; 
  } 
  return false;
}

vidType Graph::intersect_num(vidType v, vidType u, vlabel_t label) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType v_size = get_degree(v);
  vidType u_size = get_degree(u);
  vidType* v_ptr = &edges[vertices[v]];
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < v_size && idx_r < u_size) {
    vidType a = v_ptr[idx_l];
    vidType b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a == b && vlabels[a] == label) num++;
  }
  return num;
}

