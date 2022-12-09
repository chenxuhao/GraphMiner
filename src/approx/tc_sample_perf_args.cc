// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void TCSolver(Graph &g, uint64_t &total, int, int, vector<float> args) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Triangle Counting (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  uint64_t counter = 0;
  int numIters = g.V() * args[0];
  printf("num iters: %d\n", numIters);
  int skips = 0;
  #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
  for (vidType u = 0; u < numIters; u ++) {
    auto adj_u = g.N(u);
    int j = 0;
    for (auto v : adj_u) {
      if (j > std::round(args[1] * adj_u.size())) break;
      counter += (uint64_t)intersection_num(adj_u, g.N(v));
      j++;
    }
  }
  float p = args[0] * args[1];
  total = counter / p;
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}

