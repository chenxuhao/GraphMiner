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
  //std::vector<eidType> scale_accesses;
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
  vidType* adj_ptr(vidType v) { return &edges[vertices[v]]; }
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
  vidType* get_src_ptr() { return &src_list[0]; }
  vidType* get_dst_ptr() { return &dst_list[0]; }
  vidType get_src(eidType eid) { return src_list[eid]; }
  vidType get_dst(eidType eid) { return dst_list[eid]; }
  void print_graph();
  void orientation();
  bool is_connected(vidType v, vidType u);
  bool is_connected(std::vector<vidType> sg);
  bool binary_search(vidType key, eidType begin, eidType end);
  size_t init_edgelist(bool sym_break = false, bool ascend = false);
  std::vector<eidType> split_edgelist(int n, std::vector<vidType*> &src_ptrs, std::vector<vidType*> &dst_ptrs, int stride);
  inline int smallest_score_id(size_t n, int64_t* scores);
  vidType intersect_num(vidType v, vidType u, vlabel_t label);
};

