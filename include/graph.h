// This code is modified from AutoMine and GraphZero
// Daniel Mawhirter and Bo Wu. SOSP 2019.
// AutoMine: Harmonizing High-Level Abstraction and High Performance for Graph Mining

#pragma once
#include "common.h"
#include "scan.h"
#include "timer.h"
#include "VertexSet.h"

using namespace std;

constexpr bool map_edges = false; // use mmap() instead of read()
constexpr bool map_vertices = false; // use mmap() instead of read()
constexpr bool map_vlabels = false; // use mmap() instead of read()

class Graph {
private:
  vidType n_vertices, *edges;
  eidType n_edges, *vertices;
  vlabel_t *vlabels;
  elabel_t *elabels;
  vidType max_degree;
  int num_vertex_classes;
  size_t nnz; // for COO format
  vidType *src_list, *dst_list; // for COO format
  std::vector<std::vector<vidType>> srcs;
  std::vector<std::vector<vidType>> dsts;
  template<typename T>
  static void read_file(std::string fname, T *& pointer, size_t elements) {
    pointer = custom_alloc_global<T>(elements);
    assert(pointer);
    std::ifstream inf(fname.c_str(), std::ios::binary);
    if (!inf.good()) {
      std::cerr << "Failed to open file: " << fname << "\n";
      exit(1);
    }
    inf.read(reinterpret_cast<char*>(pointer), sizeof(T) * elements);
    inf.close();
  }
  template<typename T>
  static void map_file(std::string fname, T *& pointer, size_t elements) {
    int inf = open(fname.c_str(), O_RDONLY, 0);
    if (-1 == inf) {
      std::cerr << "Failed to open file: " << fname << "\n";
      exit(1);
    }
    pointer = (T*)mmap(nullptr, sizeof(T) * elements,
                       PROT_READ, MAP_SHARED, inf, 0);
    assert(pointer != MAP_FAILED);
    close(inf);
  }

public:
  Graph(std::string prefix, bool use_dag = false, bool has_vlabel = false) :
      vlabels(NULL), elabels(NULL), nnz(0) {
    VertexSet::release_buffers();
    std::ifstream f_meta((prefix + ".meta.txt").c_str());
    assert(f_meta);
    int vid_size;
    f_meta >> n_vertices >> n_edges >> vid_size >> max_degree;
    assert(sizeof(vidType) == vid_size);
    f_meta.close();
    // read row pointers
    if (map_vertices) map_file(prefix + ".vertex.bin", vertices, n_vertices+1);
    else read_file(prefix + ".vertex.bin", vertices, n_vertices+1);
    // read column indices
    if (map_edges) map_file(prefix + ".edge.bin", edges, n_edges);
    else read_file(prefix + ".edge.bin", edges, n_edges);
    // read vertex labels
    if (has_vlabel) {
      if (map_vlabels) map_file(prefix + ".vlabel.bin", vlabels, n_vertices);
      else read_file(prefix + ".vlabel.bin", vlabels, n_vertices);
      std::set<vlabel_t> labels;
      for (vidType v = 0; v < n_vertices; v++)
        labels.insert(vlabels[v]);
      num_vertex_classes = labels.size();
    }
    if (max_degree == 0 || max_degree>=n_vertices) exit(1);
    if (use_dag) orientation();
    std::cout << "max_degree: " << max_degree << "\n";
    VertexSet::MAX_DEGREE = std::max(max_degree, VertexSet::MAX_DEGREE);
  }
  ~Graph() {
    if (map_edges) munmap(edges, n_edges*sizeof(vidType));
    else custom_free(edges, n_edges);
    if (map_vertices) {
      munmap(vertices, (n_vertices+1)*sizeof(eidType));
    } else custom_free(vertices, n_vertices+1);
  }
  Graph(const Graph &)=delete;
  Graph& operator=(const Graph &)=delete;
  VertexSet N(vidType vid) {
    assert(vid >= 0);
    assert(vid < n_vertices);
    eidType begin = vertices[vid], end = vertices[vid+1];
    if (begin > end) {
      fprintf(stderr, "vertex %u bounds error: [%lu, %lu)\n", vid, begin, end);
      exit(1);
    }
    assert(end <= n_edges);
    return VertexSet(edges + begin, end - begin, vid);
  }
  vidType V() { return n_vertices; }
  eidType E() { return n_edges; }
  vidType size() { return n_vertices; }
  eidType sizeEdges() { return n_edges; }
  vidType num_vertices() { return n_vertices; }
  eidType num_edges() { return n_edges; }
  vidType get_degree(vidType v) { return vertices[v+1] - vertices[v]; }
  vidType out_degree(vidType v) { return vertices[v+1] - vertices[v]; }
  vidType get_max_degree() { return max_degree; }
  eidType edge_begin(vidType v) { return vertices[v]; }
  eidType edge_end(vidType v) { return vertices[v+1]; }
  vidType getEdgeDst(eidType e) { return edges[e]; }
  eidType* out_rowptr() { return vertices; }
  vidType* out_colidx() { return edges; }
  vlabel_t getData(vidType v) { return vlabels[v]; }
  vlabel_t getVertexData(vidType v) { return vlabels[v]; }
  vlabel_t get_vlabel(vidType v) { return vlabels[v]; }
  elabel_t getEdgeData(eidType e) { return elabels[e]; }
  int get_vertex_classes() { return num_vertex_classes; } // number of distinct vertex labels
  int get_edge_classes() { return 1; } // number of distinct edge labels
  vlabel_t* getVlabelPtr() { return vlabels; }
  elabel_t* getElabelPtr() { return elabels; }
  bool has_label() { return vlabels != NULL || elabels != NULL; }
  bool has_vlabel() { return vlabels != NULL; }
  bool has_elabel() { return elabels != NULL; }

  void print_graph() {
    std::cout << "Printing the graph: \n";
    for (vidType n = 0; n < n_vertices; n++) {
      std::cout << "vertex " << n << ": degree = " 
                << get_degree(n) << " edgelist = [ ";
      for (auto e = edge_begin(n); e != edge_end(n); e++)
        std::cout << getEdgeDst(e) << " ";
      std::cout << "]\n";
    }
  }

  void orientation() {
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
  vidType* get_src_ptr() { return &src_list[0]; }
  vidType* get_dst_ptr() { return &dst_list[0]; }
  vidType get_src(eidType eid) { return src_list[eid]; }
  vidType get_dst(eidType eid) { return dst_list[eid]; }
  size_t init_edgelist(bool sym_break = false, bool ascend = false) {
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
  inline int smallest_score_id(size_t n, int64_t* scores) {
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
  std::vector<eidType> split_edgelist(int n, std::vector<vidType*> &src_ptrs, std::vector<vidType*> &dst_ptrs, int stride) {
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
};

