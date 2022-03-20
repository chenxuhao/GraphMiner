// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "intersect.h"

void TCSolver(Graph &g, uint64_t &total, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP SIMD TC (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  uint64_t counter = 0;
  #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
  for (vidType u = 0; u < g.V(); u ++) {
    vidType u_size = g.get_degree(u);
    for (vidType v : g.N(u)) {
      vidType v_size = g.get_degree(v);
      counter += (uint64_t)SetIntersection::get_num(g.adj_ptr(u), u_size, g.adj_ptr(v), v_size);
    }
  }
  total = counter;
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}

