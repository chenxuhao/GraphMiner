#include "pattern.hh"

void Pattern::set_name() {
  auto n = num_vertices;
  auto m = num_edges;
  name_ = "";
  if (has_label) name_ += std::to_string(num_vertex_classes) + "color-";
  if (n == 3) {
    if (m == 2) name_ += "wedge";
    else name_ += "triangle";
  } else if (n == 4) {
    if (m == 3) {
      if (1) name_ += "3-star";
      else name_ = "4-path";
    } else if (m == 4) {
      if (1) name_ += "square";
      else name_ += "tailed_triangle";
    } else if (m == 5) {
      name_ += "diamond";
    } else {
      assert(m==6);
      name_ += "4-clique";
    }
  } else {
    name_ += "unknown";
  }
}

std::vector<vidType> Pattern::v_list() const {
  std::vector<vidType> vs;
  for (auto pair : adj_list) vs.push_back(pair.first);
  std::sort(vs.begin(), vs.end());
  return vs;
}

std::string Pattern::to_string(const std::vector<vlabel_t> &given_labels) const {
  if (labelling == LABELLED || labelling == PARTIALLY_LABELLED) {
    assert(given_labels.size() >= size_t(num_vertices));
    std::string res("");
    for (auto pair : adj_list) {
      auto u = pair.first;
      auto l1 = given_labels[u] == static_cast<vlabel_t>(-1)
        ? "*" : std::to_string(given_labels[u]);
      for (auto v : pair.second) {
        if (u > v) continue;
        auto l2 = given_labels[v] == static_cast<vlabel_t>(-1)
          ? "*" : std::to_string(given_labels[v]);
        res += "[";
        res += std::to_string(u) + "," + l1;
        res += "-";
        res += std::to_string(v) + "," + l2;
        res += "]";
      }
    }
    return res;
  } else {
    return to_string();
  }
}

std::string Pattern::to_string() const {
  if (labelling == LABELLED || labelling == PARTIALLY_LABELLED) {
    return to_string(vlabels);
  } else {
    std::string res("");
    for (auto pair : adj_list) {
      auto u = pair.first;
      for (auto v : pair.second) {
        if (u > v) continue;
        res += "[";
        res += std::to_string(u);
        res += "-";
        res += std::to_string(v);
        res += "]";
      }
    }
    return res;
  }
}
void Pattern::read_adj_file(std::string inputfile) {
  //std::cout << "Reading pattern graph from file: " << inputfile << "\n";
  std::ifstream query_graph(inputfile.c_str());
  std::string line;
  while (std::getline(query_graph, line)) {
    std::istringstream iss(line);
    std::vector<vidType> vs(std::istream_iterator<vidType>{iss}, std::istream_iterator<vidType>());
    vidType a, b;
    if (vs.size() == 2) {
      a = vs[0]; b = vs[1];
      adj_list[a].push_back(b);
      adj_list[b].push_back(a);
    } else if (vs.size() == 4) { // edge with labelled vertices
      labelling = LABELLED;
      vlabel_t alabel, blabel;
      a = vs[0]; b = vs[2];
      alabel = vs[1]; blabel = vs[3];
      adj_list[a].push_back(b);
      adj_list[b].push_back(a);
      //vlabels.resize(std::max({(vidType)vlabels.size(), a, b}));
      //vlabels[a-1] = alabel;
      //vlabels[b-1] = blabel;
      vlabels.resize(std::max({(vidType)vlabels.size(), a+1, b+1}));
      vlabels[a] = alabel;
      vlabels[b] = blabel;
    } else {
      throw std::invalid_argument("Input file '" + inputfile + "' error format");
    }
  }
  num_vertices = adj_list.size();
  if (num_vertices == 0)
    throw std::invalid_argument("Found 0 vertices in file '" + inputfile + "'");
  num_edges = 0;
  for (auto pair : adj_list)
    num_edges += pair.second.size();
  num_edges /= 2;

  max_degree = 0;
  for (vidType v = 0; v < num_vertices; ++v) {
    auto deg = get_degree(v);
    if (deg > max_degree)
      max_degree = deg;
  }

  // check if labelled or partially labelled
  if (utils::search(vlabels, static_cast<vlabel_t>(-1)))
    labelling = PARTIALLY_LABELLED;
  num_vertex_classes = 0;
  // read vertex labels
  if (labelling == LABELLED) {
    std::set<vlabel_t> labels;
    for (vidType v = 0; v < num_vertices; v++)
      labels.insert(vlabels[v]);
    num_vertex_classes = labels.size();
    assert(num_vertex_classes >= 1);
  }
  computeLabelsFrequency();
}

bool Pattern::is_connected(vidType u, vidType v) const {
  if (get_degree(u) < get_degree(v)) std::swap(u, v);
  int begin = 0;
  int end = get_degree(v)-1;
  while (begin <= end) {
    int mid = begin + ((end - begin) >> 1);
    auto w = get_neighbor(v, mid);
    if (w == u) return true;
    else if (w > u) end = mid - 1;
    else begin = mid + 1;
  }
  return false;
}

Pattern& Pattern::remove_edge(vidType u, vidType v) {
  //if (!is_anti_vertex(u) && !is_anti_vertex(v)) {
  //std::erase(adj_list[u], v);
  //std::erase(adj_list[v], u);
  //}
  //std::erase(anti_adj_list[u], v);
  //std::erase(anti_adj_list[v], u);
  return *this;
}

Pattern& Pattern::add_edge(vidType u, vidType v) {
  adj_list[u].push_back(v);
  adj_list[v].push_back(u);
  if (labelling == PARTIALLY_LABELLED || labelling == LABELLED) {
    // may have added a anti-vertex: in which case we need to give it a label
    if (v > num_vertices)
      vlabels.push_back(static_cast<vlabel_t>(-3)); // just some random label
  }
  return *this;
}

Pattern& Pattern::set_label(vidType u, vlabel_t l) {
  if (labelling == UNLABELLED || labelling == DISCOVER_LABELS)
    vlabels.resize(num_vertices);
  //vlabels[u-1] = l;
  vlabels[u] = l;
  labelling = l == static_cast<vlabel_t>(-1) ? PARTIALLY_LABELLED : LABELLED;
  return *this;
}

void Pattern::buildCoreTable() {
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

void Pattern::computeKCore() {
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
      int u = get_neighbor(v, j);
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

void Pattern::computeLabelsFrequency() {
  labels_frequency_.resize(num_vertex_classes+1);
  std::fill(labels_frequency_.begin(), labels_frequency_.end(), 0);
  max_label = 0;
  for (vidType v = 0; v < size(); ++v) {
    int label = int(get_vlabel(v));
    assert(label <= num_vertex_classes);
    if (label > max_label) max_label = label;
    labels_frequency_[label] += 1;
  }
  max_label_frequency_ = 0;
  for (auto element : labels_frequency_)
    if (element > max_label_frequency_)
      max_label_frequency_ = element;
}

void Pattern::BuildNLF() {
  nlf_ = new std::unordered_map<vlabel_t, int>[size()];
  for (vidType i = 0; i < size(); ++i) {
    for (vidType u : adj_list[i]) {
      auto label = get_vlabel(u);
      if (nlf_[i].find(label) == nlf_[i].end()) {
        nlf_[i][label] = 0;
      }
      nlf_[i][label] += 1;
    }
  }
}

void Pattern::print_meta_data() const {
  std::cout << "|V|: " << num_vertices << ", |E|: " << num_edges << ", |\u03A3|: " << num_vertex_classes << std::endl;
  std::cout << "Max Degree: " << max_degree << ", Max Label Frequency: " << max_label_frequency_ << std::endl;
}

