// Copyright 2023 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include "graph.h"
#include "pattern.hh"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/opadd_reducer.h>

typedef std::vector<std::vector<uint64_t>> PrivateCounters;

void cilk_kmotif(Graph &g, unsigned k, std::vector<uint64_t> &counters);

void MotifSolver(Graph &g, int k, std::vector<uint64_t> &total, int, int) {
  int num_threads = __cilkrts_get_nworkers();
  std::cout << "Cilk Motif solver (" << num_threads << " threads) ...\n";
  Timer t;
  t.Start();
  cilk_kmotif(g, k, total);
  t.Stop();
  std::cout << "runtime [cilk_base] = " << t.Seconds() << "\n";
}

void cilk_3motif(Graph &g, PrivateCounters &global_counters) {
  #pragma grainsize = 1
  cilk_for (vidType v0 = 0; v0 < g.V(); v0++) {
    auto tid = __cilkrts_get_worker_number();
    auto &counter = global_counters[tid];
    auto y0 = g.N(v0);
    auto y0f0 = bounded(y0,v0);
    for (vidType idx1 = 0; idx1 < y0.size(); idx1++) {
      auto v1 = y0.begin()[idx1];
      auto y1 = g.N(v1);
      counter[0] += difference_num(y0, y1, v1);
    }
    for (vidType idx1 = 0; idx1 < y0f0.size(); idx1++) {
      auto v1 = y0f0.begin()[idx1];
      auto y1 = g.N(v1);
      counter[1] += intersection_num(y0f0, y1, v1);
    }
  }
}

void cilk_4motif(Graph &g, PrivateCounters &global_counters) {
  #pragma grainsize = 1
  cilk_for (vidType v0 = 0; v0 < g.V(); v0++) {
    auto tid = __cilkrts_get_worker_number();
    auto &counter = global_counters[tid];
    auto y0 = g.N(v0);
    auto y0f0 = bounded(y0,v0);
    for(vidType idx1 = 0; idx1 < y0.size(); idx1++) {
      auto v1 = y0.begin()[idx1];
      auto y1 = g.N(v1);
      auto y0n1f1 = difference_set(y0, y1, v1);
      for(vidType idx2 = 0; idx2 < y0n1f1.size(); idx2++) {
        auto v2 = y0n1f1.begin()[idx2];
        auto y2 = g.N(v2);
        counter[0] += difference_num(y0n1f1, y2, v2);
      }
    }
    for(vidType idx1 = 0; idx1 < y0f0.size(); idx1++) {
      auto v1 = y0f0.begin()[idx1];
      auto y1 = g.N(v1);
      auto y0y1 = intersection_set(y0, y1);
      auto y0f0y1f1 = intersection_set(y0f0, y1, v1);
      VertexSet n0y1;
      difference_set(n0y1,y1, y0);
      VertexSet n0f0y1;
      difference_set(n0f0y1,y1, y0);
      auto y0n1 = difference_set(y0, y1);
      auto y0f0n1f1 = difference_set(y0f0, y1, v1);
      for(vidType idx2 = 0; idx2 < y0y1.size(); idx2++) {
        auto v2 = y0y1.begin()[idx2];
        auto y2 = g.N(v2);
        counter[4] += difference_num(y0y1, y2, v2);
        VertexSet n0n1y2;
        counter[2] += difference_num(difference_set(n0n1y2,y2, y0), y1);
      }
      for(vidType idx2 = 0; idx2 < y0f0y1f1.size(); idx2++) {
        auto v2 = y0f0y1f1.begin()[idx2];
        auto y2 = g.N(v2);
        counter[5] += intersection_num(y0f0y1f1, y2, v2);
      }
      for(vidType idx2 = 0; idx2 < y0n1.size(); idx2++) {
        auto v2 = y0n1.begin()[idx2];
        auto y2 = g.N(v2);
        counter[1] += difference_num(n0y1, y2);
      }
      for(vidType idx2 = 0; idx2 < y0f0n1f1.size(); idx2++) {
        auto v2 = y0f0n1f1.begin()[idx2];
        auto y2 = g.N(v2);
        counter[3] += intersection_num(n0f0y1, y2, v0);
      }
    }
  }
}

void cilk_kmotif(Graph &g, unsigned k, std::vector<uint64_t> &total) {
  std::cout << "Running Cilk " << k << "-motif solver\n";
  int num_threads = __cilkrts_get_nworkers();
  int num_patterns = num_possible_patterns[k];
  for (int pid = 0; pid < num_patterns; pid++) total[pid] = 0;
  PrivateCounters counters(num_threads);
  for (int tid = 0; tid < num_threads; tid++) {
    counters[tid].resize(num_patterns);
    for (int pid = 0; pid < num_patterns; pid++) {
      counters[tid][pid] = 0;
    }
  }
  
  if (k == 3) {
    cilk_3motif(g, counters);
  } else if (k == 4) {
    cilk_4motif(g, counters);
  } else if (k == 5) {
    //cilk_5motif(g, counters);
  } else {
    std::cout << "Not implemented yet\n";
    exit(0);
  }
  for (int pid = 0; pid < num_patterns; pid++)
    for (int tid = 0; tid < num_threads; tid++)
      total[pid] += counters[tid][pid];
}

