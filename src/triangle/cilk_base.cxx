// Copyright 2023 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/opadd_reducer.h>

void TCSolver(Graph &g, uint64_t &total, int, int) {
  int num_threads = __cilkrts_get_nworkers();
  std::cout << "Cilk TC (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  cilk::opadd_reducer<uint64_t> counter = 0;
  //#pragma grainsize = 1
  cilk_for (vidType u = 0; u < g.V(); u ++) {
    auto adj_u = g.N(u);
    for (auto i = 0; i < adj_u.size(); i ++) {
      auto v = adj_u[i];
      counter += (uint64_t)intersection_num(adj_u, g.N(v));
    }
  }
  total = counter;
  t.Stop();
  std::cout << "runtime [cilk_base] = " << t.Seconds() << " sec\n";
  return;
}

