#include "graph.h"
#include "scan.h"

Graph::Graph(std::string prefix, bool use_dag, bool directed, 
             bool use_vlabel, bool use_elabel, bool need_reverse, bool bipartite) :
    is_directed_(directed), is_bipartite(bipartite), 
    max_degree(0), n_vertices(0), n_edges(0), 
    nnz(0), max_label_frequency_(0), max_label(0),
    feat_len(0), num_vertex_classes(0), num_edge_classes(0), 
    edges(NULL), vertices(NULL), vlabels(NULL), elabels(NULL), 
    features(NULL), src_list(NULL), dst_list(NULL) {
  // parse file name
  size_t i = prefix.rfind('/', prefix.length());
  if (i != string::npos) inputfile_path = prefix.substr(0, i);
  i = inputfile_path.rfind('/', inputfile_path.length());
  if (i != string::npos) name_ = inputfile_path.substr(i+1);
  std::cout << "input file path: " << inputfile_path << ", graph name: " << name_ << "\n";

  // read meta information
  VertexSet::release_buffers();
  std::ifstream f_meta((prefix + ".meta.txt").c_str());
  assert(f_meta);
  int vid_size = 0, eid_size = 0, vlabel_size = 0, elabel_size = 0;
  if (bipartite) {
    f_meta >> n_vert0 >> n_vert1;
    n_vertices = n_vert0 + n_vert1;
  } else f_meta >> n_vertices;
  f_meta >> n_edges >> vid_size >> eid_size >> vlabel_size >> elabel_size
         >> max_degree >> feat_len >> num_vertex_classes >> num_edge_classes;
  assert(sizeof(vidType) == vid_size);
  assert(sizeof(eidType) == eid_size);
  assert(sizeof(vlabel_t) == vlabel_size);
  //assert(sizeof(elabel_t) == elabel_size);
  assert(max_degree > 0 && max_degree < n_vertices);
  f_meta.close();
  // read row pointers
  if (map_vertices) map_file(prefix + ".vertex.bin", vertices, n_vertices+1);
  else read_file(prefix + ".vertex.bin", vertices, n_vertices+1);
  // read column indices
  if (map_edges) map_file(prefix + ".edge.bin", edges, n_edges);
  else read_file(prefix + ".edge.bin", edges, n_edges);

  if (is_directed_) {
    std::cout << "This is a directed graph\n";
    if (need_reverse) {
      build_reverse_graph();
      std::cout << "This graph maintains both incomming and outgoing edge-list\n";
      has_reverse = true;
    }
  } else {
    has_reverse = true;
    reverse_vertices = vertices;
    reverse_edges = edges;
  }

  // read vertex labels
  if (use_vlabel) {
    assert (num_vertex_classes > 0);
    assert (num_vertex_classes < 255); // we use 8-bit vertex label dtype
    std::string vlabel_filename = prefix + ".vlabel.bin";
    ifstream f_vlabel(vlabel_filename.c_str());
    if (f_vlabel.good()) {
      if (map_vlabels) map_file(vlabel_filename, vlabels, n_vertices);
      else read_file(vlabel_filename, vlabels, n_vertices);
      std::set<vlabel_t> labels;
      for (vidType v = 0; v < n_vertices; v++)
        labels.insert(vlabels[v]);
      //for (int i = 0; i < n_vertices; i++) std::cout << unsigned(vlabels[i]) << "\n";
      std::cout << "# distinct vertex labels: " << labels.size() << "\n";
      assert(size_t(num_vertex_classes) == labels.size());
    } else {
      std::cout << "WARNING: vertex label file not exist; generating random labels\n";
      vlabels = new vlabel_t[n_vertices];
      for (vidType v = 0; v < n_vertices; v++) {
        vlabels[v] = rand() % num_vertex_classes + 1;
      }
    }
    auto max_vlabel = unsigned(*(std::max_element(vlabels, vlabels+n_vertices)));
    std::cout << "maximum vertex label: " << max_vlabel << "\n";
  }
  if (use_elabel) {
    std::string elabel_filename = prefix + ".elabel.bin";
    ifstream f_elabel(elabel_filename.c_str());
    if (f_elabel.good()) {
      assert (num_edge_classes > 0);
      if (map_elabels) map_file(elabel_filename, elabels, n_edges);
      else read_file(elabel_filename, elabels, n_edges);
      std::set<elabel_t> labels;
      for (eidType e = 0; e < n_edges; e++)
        labels.insert(elabels[e]);
      //for (int i = 0; i < n_edges; i++) {
      //  if (elabels[i] > 5 || elabels[i] < 1)
      //    std::cout << "elabels[" << i << "]=" << elabels[i] << "\n";
      //}
      //for (int i = 0; i < 10; i++) std::cout << elabels[i] << "\n";
      std::cout << "# distinct edge labels: " << labels.size() << "\n";
      //for (auto l : labels) std::cout << l << "\n";
      assert(size_t(num_edge_classes) >= labels.size());
    } else {
      std::cout << "WARNING: edge label file not exist; generating random labels\n";
      elabels = new elabel_t[n_edges];
      if (num_edge_classes < 1) {
        num_edge_classes = 1;
        for (eidType e = 0; e < n_edges; e++) {
          elabels[e] = 1;
        }
      } else {
        for (eidType e = 0; e < n_edges; e++) {
          elabels[e] = rand() % num_edge_classes + 1;
        }
      }
    }
    auto max_elabel = unsigned(*(std::max_element(elabels, elabels+n_edges)));
    std::cout << "maximum edge label: " << max_elabel << "\n";
  }
  // orientation: convert the undirected graph into directed. Only for k-cliques. This may change max_degree.
  if (use_dag) {
    assert(!directed); // must be undirected before orientation
    orientation();
  }
  // compute maximum degree
  VertexSet::MAX_DEGREE = std::max(max_degree, VertexSet::MAX_DEGREE);
  labels_frequency_.clear();
}

Graph::~Graph() {
  if (dst_list != NULL && dst_list != edges) delete [] dst_list;
  if (map_edges) munmap(edges, n_edges*sizeof(vidType));
  else custom_free(edges, n_edges);
  if (map_vertices) munmap(vertices, (n_vertices+1)*sizeof(eidType));
  else custom_free(vertices, n_vertices+1);
  if (vlabels != NULL) delete [] vlabels;
  if (elabels != NULL) delete [] elabels;
  if (features != NULL) delete [] features;
  if (src_list != NULL) delete [] src_list;
}

void Graph::sort_neighbors() {
  std::cout << "Sorting the neighbor lists (used for pattern mining)\n";
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    auto begin = edge_begin(v);
    auto end = edge_end(v);
    std::sort(edges+begin, edges+end);
  }
}

void Graph::build_reverse_graph() {
  std::vector<VertexList> reverse_adj_lists(n_vertices);
  for (vidType v = 0; v < n_vertices; v++) {
    for (auto u : N(v)) {
      reverse_adj_lists[u].push_back(v);
    }
  }
  reverse_vertices = custom_alloc_global<eidType>(n_vertices+1);
  reverse_vertices[0] = 0;
  for (vidType i = 1; i < n_vertices+1; i++) {
    auto degree = reverse_adj_lists[i-1].size();
    reverse_vertices[i] = reverse_vertices[i-1] + degree;
  }
  reverse_edges = custom_alloc_global<vidType>(n_edges);
  //#pragma omp parallel for
  for (vidType i = 0; i < n_vertices; i++) {
    auto begin = reverse_vertices[i];
    std::copy(reverse_adj_lists[i].begin(), 
        reverse_adj_lists[i].end(), &reverse_edges[begin]);
  }
  for (auto adjlist : reverse_adj_lists) adjlist.clear();
  reverse_adj_lists.clear();
}

VertexSet Graph::N(vidType vid) const {
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

VertexSet Graph::out_neigh(vidType vid, vidType offset) const {
  assert(vid >= 0);
  assert(vid < n_vertices);
  auto begin = vertices[vid];
  auto end = vertices[vid+1];
  if (begin > end) {
    fprintf(stderr, "vertex %u bounds error: [%lu, %lu)\n", vid, begin, end);
    exit(1);
  }
  assert(end <= n_edges);
  return VertexSet(edges + begin + offset, end - begin, vid);
}

// TODO: fix for directed graph
VertexSet Graph::in_neigh(vidType vid) const {
  assert(vid >= 0);
  assert(vid < n_vertices);
  auto begin = reverse_vertices[vid];
  auto end = reverse_vertices[vid+1];
  if (begin > end) {
    fprintf(stderr, "vertex %u bounds error: [%lu, %lu)\n", vid, begin, end);
    exit(1);
  }
  assert(end <= n_edges);
  return VertexSet(reverse_edges + begin, end - begin, vid);
}
 
void Graph::allocateFrom(vidType nv, eidType ne) {
  n_vertices = nv;
  n_edges    = ne;
  vertices = new eidType[nv+1];
  edges = new vidType[ne];
  vertices[0] = 0;
}

vidType Graph::compute_max_degree() {
  std::cout << "computing the maximum degree\n";
  Timer t;
  t.Start();
  std::vector<vidType> degrees(n_vertices, 0);
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    degrees[v] = vertices[v+1] - vertices[v];
  }
  vidType max_degree = *(std::max_element(degrees.begin(), degrees.end()));
  t.Start();
  return max_degree;
}

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

void Graph::print_graph() const {
  std::cout << "Printing the graph: \n";
  for (vidType n = 0; n < n_vertices; n++) {
    std::cout << "vertex " << n << ": degree = " 
      << get_degree(n) << " edgelist = [ ";
    for (auto e = edge_begin(n); e != edge_end(n); e++) {
      if (elabels != NULL)
        std::cout << "<";
      std::cout << getEdgeDst(e) << " ";
      if (elabels != NULL)
        std::cout << getEdgeData(e) << "> ";
    }
    std::cout << "]\n";
  }
}

eidType Graph::init_edgelist(bool sym_break, bool ascend) {
  Timer t;
  t.Start();
  if (nnz != 0) return nnz; // already initialized
  nnz = E();
  if (sym_break) nnz = nnz/2;
  sizes.resize(V());
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
      sizes[v] ++;
      i ++;
    }
  }
  //assert(i == nnz);
  t.Stop();
  std::cout << "Time on generating the edgelist: " << t.Seconds() << " sec\n";
  return nnz;
}

bool Graph::is_connected(vidType v, vidType u) const {
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

bool Graph::is_connected(std::vector<vidType> sg) const {
  return false;
}

bool Graph::binary_search(vidType key, eidType begin, eidType end) const {
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

vidType Graph::intersect_num(VertexSet& vs, vidType u, vlabel_t label) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType u_size = get_degree(u);
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < vs.size() && idx_r < u_size) {
    vidType a = vs[idx_l];
    vidType b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a == b && vlabels[a] == label) num++;
  }
  return num;
}

vidType Graph::intersect_set(vidType v, vidType u, vlabel_t label, VertexSet& result) {
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
    if (a == b && vlabels[a] == label) {
      result.add(a);
      num++;
    }
  }
  return num;
}

vidType Graph::intersect_set(VertexSet& vs, vidType u, vlabel_t label, VertexSet& result) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType u_size = get_degree(u);
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < vs.size() && idx_r < u_size) {
    vidType a = vs[idx_l];
    vidType b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a == b && vlabels[a] == label) {
      result.add(a);
      num++;
    }
  }
  return num;
}

vidType Graph::difference_num_edgeinduced(vidType v, vidType u, vlabel_t label) {
  vidType num = 0;
  vidType* v_ptr = &edges[vertices[v]];
  for (vidType i = 0; i < get_degree(v); i ++) {
    auto w = v_ptr[i];
    if (w != u && vlabels[w] == label) num++;
  }
  return num;
}

vidType Graph::difference_num_edgeinduced(VertexSet& vs, vidType u, vlabel_t label) {
  vidType num = 0;
  for (auto w : vs)
    if (w != u && vlabels[w] == label) num++;
  return num;
}

vidType Graph::difference_set_edgeinduced(vidType v, vidType u, vlabel_t label, VertexSet& result) {
  vidType num = 0;
  vidType* v_ptr = &edges[vertices[v]];
  for (vidType i = 0; i < get_degree(v); i ++) {
    auto w = v_ptr[i];
    if (w != u && vlabels[w] == label) {
      result.add(w);
      num++;
    }
  }
  return num;
}

vidType Graph::difference_set_edgeinduced(VertexSet& vs, vidType u, vlabel_t label, VertexSet& result) {
  vidType num = 0;
  for (auto w : vs) {
    if (w != u && vlabels[w] == label) {
      result.add(w);
      num++;
    }
  }
  return num;
}

vidType Graph::difference_num(vidType v, vidType u, vlabel_t label) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType v_size = get_degree(v);
  vidType u_size = get_degree(u);
  vidType* v_ptr = &edges[vertices[v]];
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < v_size && idx_r < u_size) {
    auto a = v_ptr[idx_l];
    auto b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a < b && a != u && vlabels[a] == label) num++;
  }
  while (idx_l < v_size) {
    auto a = v_ptr[idx_l];
    idx_l++;
    if (a != u && vlabels[a] == label)
      num ++;
  }
  return num;
}

vidType Graph::difference_num(VertexSet& vs, vidType u, vlabel_t label) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType u_size = get_degree(u);
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < vs.size() && idx_r < u_size) {
    auto a = vs[idx_l];
    auto b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a < b && a != u && vlabels[a] == label) num++;
  }
  while (idx_l < vs.size()) {
    auto a = vs[idx_l];
    idx_l++;
    if (a != u && vlabels[a] == label)
      num ++;
  }
  return num;
}

vidType Graph::difference_set(vidType v, vidType u, vlabel_t label, VertexSet& result) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType v_size = get_degree(v);
  vidType u_size = get_degree(u);
  vidType* v_ptr = &edges[vertices[v]];
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < v_size && idx_r < u_size) {
    auto a = v_ptr[idx_l];
    auto b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a < b && a != u && vlabels[a] == label) {
      result.add(a);
      num++;
    }
  }
  while (idx_l < v_size) {
    auto a = v_ptr[idx_l];
    idx_l++;
    if (a != u && vlabels[a] == label) {
      result.add(a);
      num ++;
    }
  }
  return num;
}

vidType Graph::difference_set(VertexSet& vs, vidType u, vlabel_t label, VertexSet& result) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType u_size = get_degree(u);
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < vs.size() && idx_r < u_size) {
    auto a = vs[idx_l];
    auto b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a < b && a != u && vlabels[a] == label) {
      result.add(a);
      num++;
    }
  }
  while (idx_l < vs.size()) {
    auto a = vs[idx_l];
    idx_l++;
    if (a != u && vlabels[a] == label) {
      result.add(a);
      num ++;
    }
  }
  return num;
}

void Graph::BuildReverseIndex() {
  if (labels_frequency_.empty()) computeLabelsFrequency();
  int nl = num_vertex_classes;
  if (max_label == num_vertex_classes) nl += 1;
  reverse_index_.resize(size());
  reverse_index_offsets_.resize(nl+1);
  reverse_index_offsets_[0] = 0;
  vidType total = 0;
  for (int i = 0; i < nl; ++i) {
    total += labels_frequency_[i];
    reverse_index_offsets_[i+1] = total;
    //std::cout << "label " << i << " frequency: " << labels_frequency_[i] << "\n";
  }
  std::vector<eidType> start(nl);
  for (int i = 0; i < nl; ++i) {
    start[i] = reverse_index_offsets_[i];
    //std::cout << "label " << i << " start: " << start[i] << "\n";
  }
  for (vidType i = 0; i < size(); ++i) {
    auto vl = vlabels[i];
    reverse_index_[start[vl]++] = i;
  }
}

#pragma omp declare reduction(vec_plus : std::vector<int> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<int>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
void Graph::computeLabelsFrequency() {
  labels_frequency_.resize(num_vertex_classes+1);
  std::fill(labels_frequency_.begin(), labels_frequency_.end(), 0);
  //max_label = int(*std::max_element(vlabels, vlabels+size()));
  #pragma omp parallel for reduction(max:max_label)
  for (int i = 0; i < size(); ++i) {
    max_label = max_label > vlabels[i] ? max_label : vlabels[i];
  }
  #pragma omp parallel for reduction(vec_plus:labels_frequency_)
  for (vidType v = 0; v < size(); ++v) {
    int label = int(get_vlabel(v));
    assert(label <= num_vertex_classes);
    labels_frequency_[label] += 1;
  }
  max_label_frequency_ = int(*std::max_element(labels_frequency_.begin(), labels_frequency_.end()));
  //std::cout << "max_label = " << max_label << "\n";
  //std::cout << "max_label_frequency_ = " << max_label_frequency_ << "\n";
  //for (size_t i = 0; i < labels_frequency_.size(); ++i)
  //  std::cout << "label " << i << " vertex frequency: " << labels_frequency_[i] << "\n";
}

int Graph::get_frequent_labels(int threshold) {
  int num = 0;
  for (size_t i = 0; i < labels_frequency_.size(); ++i)
    if (labels_frequency_[i] > threshold)
      num++;
  return num;
}

bool Graph::is_freq_vertex(vidType v, int threshold) {
  assert(v >= 0 && v < size());
  auto label = int(vlabels[v]);
  assert(label <= num_vertex_classes);
  if (labels_frequency_[label] >= threshold) return true;
  return false;
}

// NLF: neighborhood label frequency
void Graph::BuildNLF() {
  //std::cout << "Building NLF map for the data graph\n";
  nlf_.resize(size());
  #pragma omp parallel for
  for (vidType v = 0; v < size(); ++v) {
    for (auto u : N(v)) {
      auto vl = get_vlabel(u);
      if (nlf_[v].find(vl) == nlf_[v].end())
        nlf_[v][vl] = 0;
      nlf_[v][vl] += 1;
    }
  }
}

void Graph::print_meta_data() const {
  std::cout << "|V|: " << n_vertices << ", |E|: " << n_edges << ", Max Degree: " << max_degree << "\n";
  if (num_vertex_classes > 0) {
    std::cout << "vertex-|\u03A3|: " << num_vertex_classes;
    if (!labels_frequency_.empty()) 
      std::cout << ", Max Label Frequency: " << max_label_frequency_;
    std::cout << "\n";
  } else {
    std::cout  << "This graph does not have vertex labels\n";
  }
  if (num_edge_classes > 0) {
    std::cout << "edge-|\u03A3|: " << num_edge_classes << "\n";
  } else {
    std::cout  << "This graph does not have edge labels\n";
  }
  if (feat_len > 0) {
    std::cout << "Vertex feature vector length: " << feat_len << "\n";
  } else {
    std::cout  << "This graph has no input vertex features\n";
  }
}

void Graph::buildCoreTable() {
  core_table.resize(size(), 0);
  computeKCore();
  for (vidType i = 0; i < size(); ++i) {
    if (core_table[i] > 1) {
      core_length_ += 1;
    }
  }
  //for (int v = 0; v < size(); v++)
  //  std::cout << "v_" << v << " core value: " << core_table[v] << "\n";
}

void Graph::computeKCore() {
  int nv = size();
  int md = get_max_degree();
  std::vector<int> vertices(nv);          // Vertices sorted by degree.
  std::vector<int> position(nv);          // The position of vertices in vertices array.
  std::vector<int> degree_bin(md+1, 0);   // Degree from 0 to max_degree.
  std::vector<int> offset(md+1);          // The offset in vertices array according to degree.
  for (int i = 0; i < nv; ++i) {
    int degree = get_degree(i);
    core_table[i] = degree;
    degree_bin[degree] += 1;
  }
  int start = 0;
  for (int i = 0; i < md+1; ++i) {
    offset[i] = start;
    start += degree_bin[i];
  }
  for (int i = 0; i < nv; ++i) {
    int degree = get_degree(i);
    position[i] = offset[degree];
    vertices[position[i]] = i;
    offset[degree] += 1;
  }
  for (int i = md; i > 0; --i) {
    offset[i] = offset[i - 1];
  }
  offset[0] = 0;
  for (int i = 0; i < nv; ++i) {
    int v = vertices[i];
    for(int j = 0; j < get_degree(v); ++j) {
      int u = N(v, j);
      if (core_table[u] > core_table[v]) {
        // Get the position and vertex which is with the same degree
        // and at the start position of vertices array.
        int cur_degree_u = core_table[u];
        int position_u = position[u];
        int position_w = offset[cur_degree_u];
        int w = vertices[position_w];
        if (u != w) {
          // Swap u and w.
          position[u] = position_w;
          position[w] = position_u;
          vertices[position_u] = w;
          vertices[position_w] = u;
        }
        offset[cur_degree_u] += 1;
        core_table[u] -= 1;
      }
    }
  }
}

