// Copyright 2023 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include "graph.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/opadd_reducer.h>

void cilk_kclique(Graph &g, unsigned k, uint64_t &total);

void CliqueSolver(Graph &g, int k, uint64_t &total, int, int) {
  int num_threads = __cilkrts_get_nworkers();
  std::cout << "Cilk " << k << "-clique listing (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  cilk_kclique(g, k, total);
  t.Stop();
  double run_time = t.Seconds();
  std::cout << "runtime [cilk_base] = " << run_time << " sec\n";
  return;
}

void cilk_4clique(Graph &g, uint64_t &total) {
  cilk::opadd_reducer<uint64_t> counter = 0;
  #pragma grainsize = 1
  cilk_for (vidType v0 = 0; v0 < g.V(); v0++) {
    uint64_t local_counter = 0;
    auto y0 = g.N(v0);
    for (auto v1 : y0) {
      auto y0y1 = y0 & g.N(v1);
      for (auto v2 : y0y1) {
        local_counter += intersection_num(y0y1, g.N(v2));
      }
    }
    counter += local_counter;
  }
  total = counter;
}

void cilk_5clique(Graph &g, uint64_t &total) {
  cilk::opadd_reducer<uint64_t> counter = 0;
  #pragma grainsize = 1
  cilk_for (vidType v1 = 0; v1 < g.V(); v1++) {
    uint64_t local_counter = 0;
    auto y1 = g.N(v1);
    for (auto v2 : y1) {
      auto y1y2 = intersection_set(y1, g.N(v2));
      for (auto v3 : y1y2) {
        auto y1y2y3 = intersection_set(y1y2, g.N(v3));
        for (auto v4 : y1y2y3) {
          local_counter += intersection_num(y1y2y3, g.N(v4));
        }
      }
    }
    counter += local_counter;
  }
  total = counter;
}

void cilk_kclique(Graph &g, unsigned k, uint64_t &total) {
  std::cout << "Running Cilk k-clique solver\n";
  if (k == 4) {
    cilk_4clique(g, total);
  } else if (k == 5) {
    cilk_5clique(g, total);
  } else {
    std::cout << "Not implemented yet\n";
    exit(0);
  }
}

