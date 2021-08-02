// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include "graph.h"
// Motif Couting
// count the frequenncy of motifs (subgraphs/patterns/graphlets)

static int num_possible_patterns[] = {
  0,
  1,
  1,
  2, // size 3
  6, // size 4
  21, // size 5
  112, // size 6
  853, // size 7
  11117, // size 8
  261080, // size 9
};

void MotifSolver(Graph &g, int k, std::vector<uint64_t> &total, int n_gpus, int chunk_size);

static inline void update_ccodes(unsigned level, Graph &g, const VertexId u, std::vector<uint8_t> &ccodes) {
  for (auto v : g.N(u)) {
    ccodes[v] += 1 << level;
  }
}

static inline void resume_ccodes(unsigned level, Graph &g, const VertexId u, std::vector<uint8_t> &ccodes) {
  for (auto v : g.N(u)) {
    ccodes[v] -= 1 << level;
  }
}

static inline void update_ccodes(unsigned level, Graph &g, const VertexId u, std::vector<uint8_t> &ccodes, const VertexId up) {
  for (auto v : g.N(u)) {
    if (v >= up) break;
    ccodes[v] += 1 << level;
  }
}

static inline void resume_ccodes(unsigned level, Graph &g, const VertexId u, std::vector<uint8_t> &ccodes, const VertexId up) {
  for (auto v : g.N(u)) {
    if (v >= up) break;
    ccodes[v] -= 1 << level;
  }
}

// applying partial orders
static inline bool do_early_break(unsigned k, unsigned level, vidType dst, unsigned src_idx, const VertexList *emb) {
  if (k == 3) {
    if (src_idx == 0 && dst >= (*emb)[1]) return true;
    if (src_idx == 1 && dst >= (*emb)[0]) return true;
  } else {
    if (dst >= (*emb)[0]) return true;
    for (unsigned i = src_idx + 1; i < level + 1; ++i)
      if (dst >= (*emb)[i]) return true;
  }
  return false;
}

static inline bool is_connected(unsigned level, unsigned id, uint8_t ccode) {
  if (level == 1) {
    return (ccode == id + 1) || (ccode == 3);
  } else if (level == 2) {
    return (ccode & (1 << id));
  } else {
    return false;
  }
}

static inline bool is_canonical(unsigned k, unsigned level, vidType dst, unsigned src_idx, uint8_t ccode, const VertexList *emb) {
  if (k == 3) {
    if (src_idx == 1 && ccode == 3) return false;
  } else {
    for (unsigned i = 1; i < level+1; ++i)
      if (dst == (*emb)[i]) return false;
    for (unsigned i = 0; i < src_idx; ++i)
      if (is_connected(level, i, ccode))
        return false;
  }
  return true;
}

static inline unsigned get_pattern_id(unsigned level, uint8_t ccode,
                                      unsigned pcode, unsigned src_idx) {
  unsigned pid = 0;
  if (level == 1) { // count 3-motifs
    if (ccode == 3) {
      pid = 0; // triangle
    } else {
      pid = 1; // wedge
    }
  } else if (level == 2) {   // count 4-motifs
    if (pcode == 0) { // extending a triangle
      if (ccode == 7) {
        pid = 5; // clique
      } else if (ccode == 3 || ccode == 5 || ccode == 6) {
        pid = 4; // diamond
      } else
        pid = 3; // tailed-triangle
    } else {
      if (ccode == 7) {
        pid = 4; // diamond
      } else if (src_idx == 0) {
        if (ccode == 6)
          pid = 2; // 4-cycle
        else if (ccode == 3 || ccode == 5)
          pid = 3; // tailed-triangle
        else if (ccode == 1)
          pid = 1; // 3-star
        else
          pid = 0; // 4-chain
      } else {
        if (ccode == 5)
          pid = 2; // 4-cycle
        else if (ccode == 3 || ccode == 6)
          pid = 3; // tailed-triangle
        else if (ccode == 2)
          pid = 1; // 3-star
        else
          pid = 0; // 4-chain
      }
    }
  }
  return pid;
}

