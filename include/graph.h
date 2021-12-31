#pragma once
#include "VertexSet.h"

using namespace std;

constexpr bool map_edges = false; // use mmap() instead of read()
constexpr bool map_vertices = false; // use mmap() instead of read()
constexpr bool map_vlabels = false; // use mmap() instead of read()
typedef std::unordered_map<vlabel_t, int> nlf_map;

class Graph {
private:
  vidType n_vertices, *edges;
  eidType n_edges, *vertices;
  vlabel_t *vlabels;
  elabel_t *elabels;
  vidType max_degree;
  int num_vertex_classes;
  eidType nnz; // for COO format
  vidType *src_list, *dst_list; // for COO format
  VertexList reverse_index_;
  std::vector<eidType> reverse_index_offsets_;
  std::vector<vidType> labels_frequency_;
  vidType max_label_frequency_;
  int max_label;
  std::vector<nlf_map> nlf_;
  std::vector<vidType> sizes;

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
    pointer = (T*)mmap(nullptr, sizeof(T) * elements, PROT_READ, MAP_SHARED, inf, 0);
    assert(pointer != MAP_FAILED);
    close(inf);
  }
public:
  Graph(std::string prefix, bool use_dag = false, bool has_vlabel = false);
  ~Graph();
  Graph(const Graph &)=delete;
  Graph& operator=(const Graph &)=delete;
  VertexSet N(vidType vid) const;
  vidType V() const { return n_vertices; }
  eidType E() const { return n_edges; }
  eidType get_num_tasks() const { return nnz; }
  vidType size() const { return n_vertices; }
  eidType sizeEdges() const { return n_edges; }
  vidType num_vertices() const { return n_vertices; }
  eidType num_edges() const { return n_edges; }
  vidType get_degree(vidType v) const { return vertices[v+1] - vertices[v]; }
  vidType out_degree(vidType v) const { return vertices[v+1] - vertices[v]; }
  vidType get_max_degree() const { return max_degree; }
  eidType edge_begin(vidType v) const { return vertices[v]; }
  eidType edge_end(vidType v) const { return vertices[v+1]; }
  vidType* adj_ptr(vidType v) const { return &edges[vertices[v]]; }
  vidType getEdgeDst(eidType e) const { return edges[e]; }
  eidType* out_rowptr() { return vertices; }
  vidType* out_colidx() { return edges; }
  vlabel_t getData(vidType v) const { return vlabels[v]; }
  vlabel_t getVertexData(vidType v) const { return vlabels[v]; }
  vlabel_t get_vlabel(vidType v) const { return vlabels[v]; }
  elabel_t getEdgeData(eidType e) const { return elabels[e]; }
  elabel_t get_elabel(eidType e) const { return elabels[e]; }
  int get_vertex_classes() { return num_vertex_classes; } // number of distinct vertex labels
  int get_edge_classes() { return 1; } // number of distinct edge labels
  vlabel_t* getVlabelPtr() { return vlabels; }
  elabel_t* getElabelPtr() { return elabels; }
  bool has_label() { return vlabels != NULL || elabels != NULL; }
  bool has_vlabel() { return vlabels != NULL; }
  bool has_elabel() { return elabels != NULL; }
  bool is_freq_vertex(vidType v, int minsup);
  vidType* get_src_ptr() { return &src_list[0]; }
  vidType* get_dst_ptr() { return &dst_list[0]; }
  vidType get_src(eidType eid) { return src_list[eid]; }
  vidType get_dst(eidType eid) { return dst_list[eid]; }
  const nlf_map* getVertexNLF(const vidType id) const { return &nlf_[id]; }
  int get_frequent_labels(int threshold);
  int get_max_label() { return max_label; }
  int *get_label_freq_ptr() { return labels_frequency_.data(); }
  vidType getLabelsFrequency(vlabel_t label) const { return labels_frequency_.at(label); }
  const vidType* getVerticesByLabel(vlabel_t vl, vidType& count) const {
    auto start = reverse_index_offsets_[vl];
    count = reverse_index_offsets_[vl+1] - start;
    return &reverse_index_[start];
  }
  void print_graph() const;
  void print_meta_data() const;
  void orientation();
  bool is_connected(vidType v, vidType u) const;
  bool is_connected(std::vector<vidType> sg) const;
  bool binary_search(vidType key, eidType begin, eidType end) const;
  eidType init_edgelist(bool sym_break = false, bool ascend = false);
  vidType intersect_num(vidType v, vidType u, vlabel_t label);
  vidType get_max_label_frequency() const { return max_label_frequency_; }
  std::vector<vidType> get_sizes() const { return sizes; }
  void BuildNLF();
  void BuildReverseIndex();
  void computeLabelsFrequency();
};

