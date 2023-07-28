#pragma once
#include "VertexSet.h"

using namespace std;

// for out-of-core solution, enable mmap flags below
constexpr bool map_edges = false;
constexpr bool map_vertices = false;
constexpr bool map_vlabels = false;
constexpr bool map_elabels = false;
constexpr bool map_features = false;

class Graph {
protected:
  std::string name_;            // name of the graph
  std::string inputfile_path;   // file path of the graph
  bool is_directed_;            // is it a directed graph?
  bool is_bipartite;            // is it a bipartite graph?
  bool has_reverse;             // has reverse/incoming edges maintained
  vidType max_degree;           // maximun degree
  vidType n_vertices;           // number of vertices
  eidType n_edges;              // number of edges
  eidType nnz;                  // number of edges in COO format (may be halved due to orientation)
  vidType max_label_frequency_; // maximum label frequency
  vidType n_vert0;              // number of type 0 vertices for bipartite graph
  vidType n_vert1;              // number of type 1 vertices for bipartite graph
  int max_label;                // maximum label
  int feat_len;                 // vertex feature vector length: '0' means no features
  int num_vertex_classes;       // number of distinct vertex labels: '0' means no vertex labels
  int num_edge_classes;         // number of distinct edge labels: '0' means no edge labels
  int core_length_;

  vidType *edges;               // column indices of CSR format
  eidType *vertices;            // row pointers of CSR format
  vidType *reverse_edges;       // reverse column indices of CSR format
  eidType *reverse_vertices;    // reverse row pointers of CSR format
  vlabel_t *vlabels;            // vertex labels
  elabel_t *elabels;            // edge labels
  feat_t *features;             // vertex features; one feature vector per vertex
  vidType *src_list, *dst_list; // source and destination vertices of COO format
  std::vector<int> core_table;  // coreness for each vertex
  VertexList labels_frequency_; // vertex count of each label
  VertexList sizes;             // neighbor count of each source vertex in the edgelist
  VertexList reverse_index_;    // indices to vertices grouped by vertex label
  std::vector<nlf_map> nlf_;    // neighborhood label frequency
  std::vector<eidType> reverse_index_offsets_; // pointers to each vertex group

public:
  Graph(std::string prefix, bool use_dag = false, bool directed = false,
        bool use_vlabel = false, bool use_elabel = false, 
        bool need_reverse = false, bool bipartite = false);
  Graph() : name_(""), n_vertices(0), n_edges(0), nnz(0), 
            max_label_frequency_(0), max_label(0), feat_len(0), 
            num_vertex_classes(0), num_edge_classes(0), core_length_(0),
            edges(NULL), vertices(NULL), vlabels(NULL), elabels(NULL),
            features(NULL), src_list(NULL), dst_list(NULL) { }
  Graph(vidType nv, eidType ne) { allocateFrom(nv, ne); }
  ~Graph();
  Graph(const Graph &)=delete;
  Graph& operator=(const Graph &)=delete;

  // get methods for graph meta information
  vidType V() const { return n_vertices; }
  eidType E() const { return n_edges; }
  vidType V(int type) const { if (type == 0) return n_vert0; else return n_vert1; }
  eidType get_num_tasks() const { return nnz; }
  vidType num_vertices() const { return n_vertices; }
  eidType num_edges() const { return n_edges; }
  std::string get_name() const { return name_; }
  std::string get_inputfile_path() const { return inputfile_path; }
  bool is_directed() const { return is_directed_; }
  bool has_reverse_graph() const { return has_reverse; }
  vidType get_max_degree() const { return max_degree; }

  // get methods for graph topology information
  vidType get_degree(vidType v) const { return vertices[v+1] - vertices[v]; }
  vidType out_degree(vidType v) const { return vertices[v+1] - vertices[v]; }
  eidType edge_begin(vidType v) const { return vertices[v]; }
  eidType edge_end(vidType v) const { return vertices[v+1]; }
  vidType* adj_ptr(vidType v) const { return &edges[vertices[v]]; }
  vidType N(vidType v, vidType n) const { return edges[vertices[v]+n];} // get the n-th neighbor of v
  VertexSet N(vidType v) const;              // get the neighbor list of vertex v
  eidType* out_rowptr() { return vertices; } // get row pointers array
  vidType* out_colidx() { return edges; }    // get column indices array
  eidType* in_rowptr() { return reverse_vertices; } // get incoming row pointers array
  vidType* in_colidx() { return reverse_edges; }    // get incoming column indices array
  bool is_connected(vidType v, vidType u) const; // is vertex v and u connected by an edge
  bool is_connected(std::vector<vidType> sg) const; // is the subgraph sg a connected one
  VertexSet out_neigh(vidType v, vidType off = 0) const; // get the outgoing neighbor list of vertex v
  VertexSet in_neigh(vidType v) const;              // get the ingoing neighbor list of vertex v
  void build_reverse_graph();

  // Galois compatible APIs
  vidType size() const { return n_vertices; }
  eidType sizeEdges() const { return n_edges; }
  vidType getEdgeDst(eidType e) const { return edges[e]; } // get target vertex of the edge e
  vlabel_t getData(vidType v) const { return vlabels[v]; }
  vlabel_t getVertexData(vidType v) const { return vlabels[v]; }
  elabel_t getEdgeData(eidType e) const { return elabels[e]; }
  void fixEndEdge(vidType vid, eidType row_end) { vertices[vid + 1] = row_end; }
  void allocateFrom(vidType nv, eidType ne);
  void constructEdge(eidType eid, vidType dst) { edges[eid] = dst; }

  // get methods for labels and coreness
  vlabel_t get_vlabel(vidType v) const { return vlabels[v]; }
  elabel_t get_elabel(eidType e) const { return elabels[e]; }
  int get_vertex_classes() { return num_vertex_classes; } // number of distinct vertex labels
  int get_edge_classes() { return num_edge_classes; } // number of distinct edge labels
  int get_frequent_labels(int threshold);
  int get_max_label() { return max_label; }
  vlabel_t* getVlabelPtr() { return vlabels; }
  elabel_t* getElabelPtr() { return elabels; }
  vlabel_t* get_vlabel_ptr() { return vlabels; }
  elabel_t* get_elabel_ptr() { return elabels; }
  bool has_label() { return vlabels != NULL || elabels != NULL; }
  bool has_vlabel() { return vlabels != NULL; }
  bool has_elabel() { return elabels != NULL; }
  int getCoreValue(const vidType vid) const { return core_table[vid]; }
  int get2CoreSize() const { return core_length_; }

  // edgelist or COO
  vidType* get_src_ptr() { return &src_list[0]; }
  vidType* get_dst_ptr() { return &dst_list[0]; }
  vidType get_src(eidType eid) { return src_list[eid]; }
  vidType get_dst(eidType eid) { return dst_list[eid]; }
  std::vector<vidType> get_sizes() const { return sizes; }
  eidType init_edgelist(bool sym_break = false, bool ascend = false);

  // build auxiliary structures for vertex label frequency
  void BuildNLF();               // NLF: neighborhood label frequency
  void BuildReverseIndex();      // reverse index
  void computeLabelsFrequency();
  void buildCoreTable();
  void computeKCore();
  void sort_neighbors();
  bool is_freq_vertex(vidType v, int minsup);
  vidType get_max_label_frequency() const { return max_label_frequency_; }
  const nlf_map* getVertexNLF(const vidType id) const { return &nlf_[id]; }
  int *get_label_freq_ptr() { return labels_frequency_.data(); }
  vidType getLabelsFrequency(vlabel_t label) const { return labels_frequency_.at(label); }
  const vidType* getVerticesByLabel(vlabel_t vl, vidType& count) const {
    auto start = reverse_index_offsets_[vl];
    count = reverse_index_offsets_[vl+1] - start;
    return &reverse_index_[start];
  }

  // edge orientation: convert the graph from undirected to directed
  void orientation();
  vidType intersect_num(vidType v, vidType u, vlabel_t label);
  vidType intersect_num(VertexSet& vs, vidType u, vlabel_t label);
  vidType intersect_set(vidType v, vidType u, vlabel_t label, VertexSet& result);
  vidType intersect_set(VertexSet& vs, vidType u, vlabel_t label, VertexSet& result);
  vidType difference_num(vidType v, vidType u, vlabel_t label);
  vidType difference_num(VertexSet& vs, vidType u, vlabel_t label);
  vidType difference_set(vidType v, vidType u, vlabel_t label, VertexSet& result);
  vidType difference_set(VertexSet& vs, vidType u, vlabel_t label, VertexSet& result);
  vidType difference_num_edgeinduced(vidType v, vidType u, vlabel_t label);
  vidType difference_num_edgeinduced(VertexSet& vs, vidType u, vlabel_t label);
  vidType difference_set_edgeinduced(vidType v, vidType u, vlabel_t label, VertexSet& result);
  vidType difference_set_edgeinduced(VertexSet& vs, vidType u, vlabel_t label, VertexSet& result);

  // print graph information
  void print_meta_data() const;
  void print_graph() const;

 protected:
  vidType compute_max_degree();
  bool binary_search(vidType key, eidType begin, eidType end) const;
};

typedef Graph BipartiteGraph;
