// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu> and Anna Arpaci-Dusseau <annaad@mit.edu>
#include "graph.h"

#define P .1

void TCSolver(Graph &g, uint64_t &total, int, int) {
  assert(USE_DAG);
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP 4-Clique Counting (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  uint64_t counter = 0;
  int numIters = g.V() * P;
  #pragma omp parallel for schedule(dynamic, 1) reduction(+:counter)
  for (vidType v0 = 0; v0 < numIters; v0++) {
    uint64_t local_counter = 0;
    //auto tid = omp_get_thread_num();
    auto y0 = g.N(v0);
    int iter = 0;
    for (auto v1 : y0) {
      if(iter > P * y0.size()) break;
      int j = 0;
      auto y0y1 = y0 & g.N(v1);
      for (auto v2 : y0y1) {
        if(j > P * y0y1.size()) break;
        float p = P * P * P;
        local_counter += intersection_num(y0y1, g.N(v2)) / p;
        j += 1;
      }
      iter += 1;
    }
    counter += local_counter;
  }
  total = counter;
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}


