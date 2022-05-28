#pragma once
#include "graph.h"

class EmbList {
public:
  EmbList() {}
  EmbList(unsigned k, vidType l, int np) {
    init(k, l, np);
  }
  ~EmbList() {}
  void init(unsigned k, vidType l, int np=1) {
    vid_lists.resize(k-2);
    num_emb.resize(k-2);
    for (unsigned i = 0; i < k-2; i++) {
      vid_lists[i].resize(l);
      num_emb[i].resize(1);
    }
    max_level = k;
    max_length = l;
    if (np > 1 && k > 3) {
      src_indices.resize(k-3);
      if (k == 4) {
        vid_lists[1].resize(l*2);
        src_indices[0].resize(l*2);
        num_emb[1].resize(2);
      } else if (k == 5) {
        vid_lists[1].resize(l*2);
        vid_lists[2].resize(l*6);
        num_emb[1].resize(2);
        num_emb[2].resize(6);
        src_indices[0].resize(l*2);
        src_indices[1].resize(l*6);
      } else {
        std::cerr << "Not supported yet\n";
        exit(0);
      }
    }
  }
  void add_emb(unsigned level, vidType v) {
    assert(level >= 1);
    auto start = num_emb[level-1][0];
    if (start >= max_length) {
      std::cout << "v " << v << " start " << start << " level " << level << "\n";
      exit(1);
    }
    assert(start < max_length);
    vid_lists[level-1][start] = v;
    num_emb[level-1][0] ++;
  }
  void add_emb(unsigned level, vidType v, int pid, unsigned src_idx) {
    assert(level >= 2);
    auto start = num_emb[level-1][pid];
    vid_lists[level-1][pid*max_length+start] = v;
    num_emb[level-1][pid] ++;
    src_indices[level-2][pid*max_length+start] = src_idx;
  }
  uint8_t get_src(unsigned level, int idx, int pid) { 
    return src_indices[level-2][pid*max_length+idx];
  }
  VertexList & get_history() { return history; }
  vidType get_history(unsigned level) const { return history[level]; }
  const std::vector<vidType>* get_history_ptr() const { return &history; }
  void push_history(vidType vid) { history.push_back(vid); }
  void pop_history() { history.pop_back(); }
  void clean_history() { history.clear(); }
  vidType size(unsigned level) const { return num_emb[level-1][0]; }
  vidType size(unsigned level, int pid) const { return num_emb[level-1][pid]; }
  void set_size(unsigned level, vidType size) { num_emb[level-1][0] = size; }
  void set_size(unsigned level, vidType size, int pid) { num_emb[level-1][pid] = size; }
  void clear_size(unsigned level) {
    for (size_t i = 0; i < num_emb[level-1].size(); i++)
      num_emb[level-1][i] = 0;
  }
  vidType get_vertex(unsigned level, vidType i) const {
    assert(level >= 1);
    return vid_lists[level-1][i];
  }
  vidType get_vertex(unsigned level, vidType i, int pid) const {
    assert(level >= 1);
    return vid_lists[level-1][pid*max_length+i];
  }
  std::string to_string() const {
		std::stringstream ss;
    ss << "(";
    for (unsigned i = 0; i < history.size() - 1; ++i)
      ss << history[i] << ", ";
    ss << history[history.size() - 1];
    ss << ")";
		return ss.str();
  }

private:
  unsigned max_level;
  vidType max_length;
  VertexList history;
  std::vector<std::vector<vidType>> num_emb; // number of embeddings per level per pattern
  std::vector<VertexList> vid_lists; // list of vertex IDs
  //std::vector<PidList> pid_lists;   // pid[i] is the pattern id of each embedding
  std::vector<std::vector<uint8_t>> src_indices; // list of source indices
};
