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
  int step = rand();
  int seed = rand();
  #pragma omp parallel for schedule(dynamic, 1) reduction(+:counter)
  for (vidType v0 = 0; v0 < numIters; v0++) {
    uint64_t local_counter = 0;
    //auto tid = omp_get_thread_num();
    auto y0 = g.N(abs((v0*step + seed) % g.V()));
    int iter = 0;
    for (int i = 0; i < y0.size()*args[1]; i++) {
      auto v1 = y0[abs((i*step + seed) % y0.size())];
      if(iter > args[1] * y0.size()) break;
      auto y0y1 = y0 & g.N(v1);
      for (int j = 0; j < y0y1.size()*args[2]; j++) {
        auto v2 = y0y1[abs((j*step + seed) % y0y1.size())];
        local_counter += intersection_num(y0y1, g.N(v2));
      }
    }
    counter += local_counter;
  }
  float p = args[0] * args[1] * args[2];
  total = counter / p;
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}


