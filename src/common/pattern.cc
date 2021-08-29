#include "pattern.hh"

void Pattern::set_name() {
  auto n = num_vertices();
  auto m = num_edges();
  if (n == 3) {
    if (m == 2) name_ = "wedge";
    else name_ = "triangle";
  } else if (n == 4) {
    if (m == 3) {
      if (1) name_ = "3-star";
      else name_ = "4-path";
    } else if (m == 4) {
      if (1) name_ = "tailed_triangle";
      else name_ = "4-cycle";
    } else if (m == 5) {
      name_ = "diamond";
    } else {
      assert(m==6);
      name_ = "4-clique";
    }
  } else {
    name_ = "unknown";
  }
}

int32_t Pattern::num_edges() const {
    int32_t count = 0;
    for (const auto &[u, nbrs] : adj_list)
      count += nbrs.size();
    return count / 2;
  }
  
std::vector<vidType> Pattern::v_list() const {
    std::vector<vidType> vs;
    for (const auto &[v, _] : adj_list) vs.push_back(v);
    std::sort(vs.begin(), vs.end());
    return vs;
  }

std::string Pattern::to_string(const std::vector<uint32_t> &given_labels) const {
  if (labelling == LABELLED || labelling == PARTIALLY_LABELLED) {
    assert(given_labels.size() >= size_t(num_vertices()));
    std::string res("");
    for (const auto &[u, nbrs] : adj_list) {
      auto l1 = given_labels[u-1] == static_cast<uint32_t>(-1)
        ? "*" : std::to_string(given_labels[u-1]);
      for (auto v : nbrs) {
        if (u > v) continue;
        auto l2 = given_labels[v-1] == static_cast<uint32_t>(-1)
          ? "*" : std::to_string(given_labels[v-1]);
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
    return to_string(labels);
  } else {
    std::string res("");
    for (const auto &[u, nbrs] : adj_list) {
      for (auto v : nbrs) {
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
  std::cout << "Reading pattern graph from file: " << inputfile << "\n";
  std::ifstream query_graph(inputfile.c_str());
  std::string line;
  while (std::getline(query_graph, line)) {
    std::istringstream iss(line);
    std::vector<uint32_t> vs(std::istream_iterator<uint32_t>{iss}, std::istream_iterator<uint32_t>());
    uint32_t a, b;
    if (vs.size() == 2) {
      a = vs[0]; b = vs[1];
      adj_list[a].push_back(b);
      adj_list[b].push_back(a);
    } else if (vs.size() == 3) { // anti-edge
      a = vs[0]; b = vs[1];
      //anti_adj_list[a].push_back(b);
      //anti_adj_list[b].push_back(a);
    } else if (vs.size() == 4) { // edge with labelled vertices
      labelling = LABELLED;
      uint32_t alabel, blabel;
      a = vs[0]; b = vs[2];
      alabel = vs[1]; blabel = vs[3];
      adj_list[a].push_back(b);
      adj_list[b].push_back(a);
      labels.resize(std::max({(uint32_t)labels.size(), a, b}));
      labels[a-1] = alabel;
      labels[b-1] = blabel;
    } else if (vs.size() == 5) { // anti-edge with labelled vertices
      labelling = LABELLED;
      uint32_t alabel, blabel;
      a = vs[0]; b = vs[2];
      alabel = vs[1]; blabel = vs[3];
      //anti_adj_list[a].push_back(b);
      //anti_adj_list[b].push_back(a);
      labels.resize(std::max({(uint32_t)labels.size(), a, b}));
      labels[a-1] = alabel;
      labels[b-1] = blabel;
    }
  }
  if (num_vertices() == 0) {
    throw std::invalid_argument("Found 0 vertices in file '" + inputfile + "'");
  }
  // check if labelled or partially labelled
  if (utils::search(labels, static_cast<uint32_t>(-1))) {
    labelling = PARTIALLY_LABELLED;
  }
  // make sure anti_adj_list.at() doesn't fail
  //for (uint32_t v = 1; v <= num_vertices(); ++v) anti_adj_list[v];
}

