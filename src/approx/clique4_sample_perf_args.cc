// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu> and Anna Arpaci-Dusseau <annaad@mit.edu>
#include "graph.h"


void TCSolver(Graph &g, uint64_t &total, int, int, vector<float> args) {
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
  int numIters = g.V() * args[0];
  #pragma omp parallel for schedule(dynamic, 1) reduction(+:counter)
  for (vidType v0 = 0; v0 < numIters; v0++) {
    uint64_t local_counter = 0;
    //auto tid = omp_get_thread_num();
    auto y0 = g.N(v0);
    int iter = 0;
    for (auto v1 : y0) {
      if(iter > args[1] * y0.size()) break;
      int j = 0;
      auto y0y1 = y0 & g.N(v1);
      for (auto v2 : y0y1) {
        if(j > args[2] * y0y1.size()) break;
        local_counter += intersection_num(y0y1, g.N(v2));
        j += 1;
      }
      iter += 1;
    }
    counter += local_counter;
  }
  float p = args[0] * args[1] * args[2];
  total = counter / p;
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}


