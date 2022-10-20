// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu> and Anna Arpaci-Dusseau <annaad@mit.edu>
#include "graph.h"
#define P 1

void EdgeSample(Graph &g) {
  std::cout << "|e| before sampling " << g.E() << "\n";
  g.edge_sparsify(P);
  std::cout <<  "|e| after sampling " << g.E() << "\n";
}

void TCSolver(Graph &g, uint64_t &total, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Triangle Counting (" << num_threads << " threads)\n";
  EdgeSample(g);
  std::cout << "Sparsified Graph\n";
  Timer t;
  t.Start();
  uint64_t counter = 0;
  #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
  for (vidType u = 0; u < g.V(); u ++) {
    auto adj_u = g.N(u);
    for (auto v : adj_u) {
      counter += (uint64_t)intersection_num(adj_u, g.N(v));
    }
  }
  total = counter * (1/(P * P * P));
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}


