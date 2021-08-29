#pragma once
#include "utils.h"

enum Labelling {
  UNLABELLED,
  LABELLED,
  PARTIALLY_LABELLED,
  DISCOVER_LABELS
};

class Pattern {
private:
  std::string name_;
  bool has_label;
  std::unordered_map<vidType, std::vector<vidType>> adj_list;
  std::vector<uint32_t> labels;
  Labelling labelling = UNLABELLED;
  int num_labels;

public:
  Pattern() : Pattern("triangle", false) { }
  Pattern(std::string name) : Pattern(name, false) { }
  Pattern(std::string filename, bool is_labeled) : 
      name_("triangle"), has_label(is_labeled) {
    read_adj_file(filename);
    if (has_label) labelling = LABELLED;
    set_name();
    num_labels = 0;
  }
  ~Pattern() {}
  bool is_wedge() { return name_ == "wedge"; }
  bool is_triangle() { return name_ == "triangle"; }
  bool is_diamond() { return name_ == "diamond"; }
  bool is_rectangle() { return name_ == "rectangle"; }
  bool is_pentagon() { return name_ == "pentagon"; }
  bool is_house() { return name_ == "house"; }
  std::string get_name() { return name_; }
  void set_name();
  vidType num_vertices() const { return adj_list.size(); }
  uint32_t label(uint32_t qv) const { return labels[qv-1]; }
  uint32_t get_label(vidType v) const { return labels[v]; }
  const std::vector<uint32_t> &get_labels() const { return labels; }
  void set_labelling(Labelling l) { labelling = l; }
  Labelling get_labelling() const { return labelling; }
  int32_t num_edges() const;
  std::vector<vidType> v_list() const;
  std::string to_string(const std::vector<uint32_t> &given_labels) const;
  std::string to_string() const;
  //std::string to_string() { return name_; }
  void read_adj_file(std::string inputfile);
  int get_num_labels() {
    if (labelling == UNLABELLED) return 0;
    if (num_labels == 0) {
      std::set<vlabel_t> unique_vlabels;
      for (vidType v = 0; v < num_vertices(); v++)
        unique_vlabels.insert(get_label(v));
      num_labels = unique_vlabels.size();
    }
    assert(num_labels >= 1);
    return num_labels;
  }
  friend std::ostream &operator <<(std::ostream &os, const Pattern &p) {
    os << p.to_string();
    return os;
  }
  const std::vector<vidType> &get_neighbours(vidType v) const {
    return adj_list.at(v);
  }
  Pattern &add_edge(vidType u, vidType v) {
    adj_list[u].push_back(v);
    adj_list[v].push_back(u);
    if (labelling == PARTIALLY_LABELLED || labelling == LABELLED) {
      // may have added a anti-vertex: in which case we need to give it a label
      if (v > num_vertices())
        labels.push_back(static_cast<uint32_t>(-3)); // just some random label
    }
    return *this;
  }
  Pattern &set_label(uint32_t u, uint32_t l) {
    //if (labelling == UNLABELLED || labelling == DISCOVER_LABELS)
    //  labels.resize(num_vertices() + num_anti_vertices());
    labels[u-1] = l;
    labelling = l == static_cast<uint32_t>(-1) ? PARTIALLY_LABELLED : LABELLED;
    return *this;
  }
  Pattern &remove_edge(vidType u, vidType v) {
    //if (!is_anti_vertex(u) && !is_anti_vertex(v)) {
      //std::erase(adj_list[u], v);
      //std::erase(adj_list[v], u);
    //}
    //std::erase(anti_adj_list[u], v);
    //std::erase(anti_adj_list[v], u);
    return *this;
  }
  // checks labels: it is assumed that the two patterns have the same structure
  bool operator==(const Pattern &p) const {
    return p.labels == labels;
  }
  bool operator<(const Pattern &p) const {
    return p.labels < labels;
  }
};

