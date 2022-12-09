// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu> and Anna Arpaci-Dusseau <annaad@mit.edu>
#include "graph.h"

#define P 0.01

void TCSolver(Graph &g, uint64_t &total, int, int) {
  assert(USE_DAG);
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP 5-Clique Counting (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  uint64_t counter = 0;
  #pragma omp parallel for schedule(dynamic, 1) reduction(+:counter)
  for (vidType v1 = 0; v1 < g.V(); v1++) { //assumes DAG
    if(!(rand() < P * RAND_MAX)) continue;
    
    //auto tid = omp_get_thread_num();
    uint64_t local_counter = 0;
    auto y1 = g.N(v1);
    for (auto v2 : y1) {
      if(!(rand() < P * RAND_MAX)) continue;
      auto y1y2 = intersection_set(y1, g.N(v2));
      for (auto v3 : y1y2) {
        if(!(rand() <  P * RAND_MAX)) continue;
        auto y1y2y3 = intersection_set(y1y2, g.N(v3));
        for (auto v4 : y1y2y3) {
          if(!(rand() < P * RAND_MAX)) continue;
          float p = (P * y1y2.size()) / y1y2.size()  * (P * y1y2y3.size()) / y1y2y3.size() *(P * y1.size()) / y1.size()  * (P * g.V()) / g.V();
          local_counter += intersection_num(y1y2y3, g.N(v4)) / p;
        }
      }
    }
    counter += local_counter;
  }
  total = counter;
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}


