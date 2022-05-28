// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include "graph.h"
#include "emb_list.h"
#include "cmap_base.h"
#include "automine_base.h"

void extend_motif(unsigned level, unsigned k, Graph &g, EmbList &emb_list, 
                  std::vector<uint8_t> &ccodes, std::vector<uint64_t> &counter);

void kmotif(Graph &g, unsigned k, std::vector<std::vector<uint64_t>> &counters,
            std::vector<EmbList> &emb_lists, 
            std::vector<std::vector<uint8_t>> &ccodes);

void MotifSolver(Graph &g, int k, std::vector<uint64_t> &total, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Motif solver (" << num_threads << " threads) ...\n";

  int num_patterns = num_possible_patterns[k];
  std::vector<std::vector<uint64_t>> global_counters(num_threads);
  std::vector<EmbList> emb_lists(num_threads);
  std::vector<std::vector<uint8_t>> ccodes(num_threads);
  auto max_degree = g.get_max_degree();
  for (int tid = 0; tid < num_threads; tid ++) {
    auto &local_counters = global_counters[tid];
    local_counters.resize(num_patterns);
    std::fill(local_counters.begin(), local_counters.end(), 0);
    auto &local_ccodes = ccodes[tid];
    local_ccodes.resize(g.size()); // the connectivity code
    std::fill(local_ccodes.begin(), local_ccodes.end(), 0);
    auto &emb_list = emb_lists[tid];
    emb_list.init(k, max_degree, num_patterns);
  }

  Timer t;
  t.Start();
  //kmotif(g, k, global_counters, emb_lists, ccodes);
  ccode_kmotif(g, k, global_counters, ccodes);
  for (int tid = 0; tid < num_threads; tid++)
    for (int pid = 0; pid < num_patterns; pid++)
      total[pid] += global_counters[tid][pid];
  t.Stop();
  std::cout << "runtime [omp_cmap] = " << t.Seconds() << "\n";
}

void extend_motif(unsigned level, unsigned k, Graph &g, EmbList &emb_list, 
                  std::vector<uint8_t> &ccodes, std::vector<uint64_t> &counter) {
  if (level == k - 2) {
    for (int pcode = 0; pcode < num_possible_patterns[level+1]; pcode++) {
      for (vidType emb_id = 0; emb_id < emb_list.size(level, pcode); emb_id++) {
        auto v = emb_list.get_vertex(level, emb_id, pcode);
        emb_list.push_history(v);
        update_ccodes(level, g, v, ccodes);
        uint8_t src_idx = 0;
        if (k > 3) src_idx = emb_list.get_src(level, emb_id, pcode);
        for (unsigned id = 0; id < level+1; id++) {
          auto src   = emb_list.get_history(id);
          auto begin = g.edge_begin(src);
          auto end = g.edge_end(src);
          for (auto e = begin; e < end; e++) {
            auto dst = g.getEdgeDst(e);
            auto emb_ptr = emb_list.get_history_ptr();
            if (do_early_break(k, level, dst, id, emb_ptr)) break;
            auto ccode = ccodes[dst];
            if (is_canonical(k, level, dst, id, ccode, emb_ptr)) {
              auto pid = get_pattern_id(level, ccode, pcode, src_idx);
              counter[pid] ++;
            }
          }
        }
        resume_ccodes(level, g, v, ccodes);
        emb_list.pop_history();
      }
    }
    return;
  }
  for (int pcode = 0; pcode < num_possible_patterns[level+1]; pcode++) {
    for (vidType emb_id = 0; emb_id < emb_list.size(level, pcode); emb_id++) {
      auto v = emb_list.get_vertex(level, emb_id, pcode);
      emb_list.push_history(v);
      update_ccodes(level, g, v, ccodes);
      emb_list.clear_size(level+1);
      for (unsigned idx = 0; idx < level+1; idx++) {
        auto src   = emb_list.get_history(idx);
        auto begin = g.edge_begin(src);
        auto end = g.edge_end(src);
        for (auto e = begin; e < end; e++) {
          auto dst = g.getEdgeDst(e);
          auto emb_ptr = emb_list.get_history_ptr();
          if (do_early_break(k, level, dst, idx, emb_ptr)) break;
          uint8_t pcode = 0;
          auto ccode = ccodes[dst];
          if (is_canonical(k, level, dst, idx, ccode, emb_ptr)) {
            auto pid = get_pattern_id(level, ccode, pcode, idx);
            emb_list.add_emb(level+1, dst, pid, idx);
          }
        }
      }
      extend_motif(level+1, k, g, emb_list, ccodes, counter);
      resume_ccodes(level, g, v, ccodes);
      emb_list.pop_history();
    }
  }
}
 
void kmotif(Graph &g, unsigned k, std::vector<std::vector<uint64_t>> &counters,
            std::vector<EmbList> &emb_lists, 
            std::vector<std::vector<uint8_t>> &ccodes) {
  #pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    auto &counter = counters.at(tid);
    auto &local_ccodes = ccodes[tid];
    auto &emb_list = emb_lists[tid];
    #pragma omp for schedule(dynamic, 1) nowait
    for (vidType v = 0; v < g.size(); v ++) {
      emb_list.clear_size(1);
      auto begin = g.edge_begin(v);
      auto end = g.edge_end(v);
      for (auto e = begin; e < end; e++) {
        auto dst = g.getEdgeDst(e);
        local_ccodes[dst] = 1;
        if (dst < v) emb_list.add_emb(1, dst);
      }
      emb_list.push_history(v);
      extend_motif(1, k, g, emb_list, local_ccodes, counter);
      for (auto e = begin; e < end; e++) {
        auto dst = g.getEdgeDst(e);
        local_ccodes[dst] = 0;
      }
      emb_list.pop_history();
    }
  }
}

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
