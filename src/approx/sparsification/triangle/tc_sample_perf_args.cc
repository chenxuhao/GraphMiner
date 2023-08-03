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
  int seed = rand();
  int step = rand();
  int numIters = g.V() * args[0];
  printf("num iters: %d\n", numIters);
  #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
  for (vidType u = 0; u < numIters; u ++) {
    auto adj_u = g.N(abs((u*step + seed) % g.V()));
    for (int v = 0; v < adj_u.size() * args[1]; v++) {
      vidType vi = adj_u[abs((v*step + seed) % adj_u.size())];
      counter += (uint64_t)intersection_num(adj_u, g.N(vi));
    }
  }
  float p = args[0] * args[1];
  total = counter / p;
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}

