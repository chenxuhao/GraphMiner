// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu> and Anna Arpaci-Dusseau <annaad@mit.edu>
#include "graph.h"

void EdgeSample(Graph &g, float p) {
  std::cout << "|e| before sampling " << g.E() << "\n";
  g.edge_sparsify(p);
  std::cout <<  "|e| after sampling " << g.E() << "\n";
}

void TCSolver(Graph &g, uint64_t &total, int, int, vector<float> args) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Triangle Counting (" << num_threads << " threads)\n";
  EdgeSample(g, args[0]);
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
  total = counter * (1/(args[0] * args[0] * args[0]));
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}


