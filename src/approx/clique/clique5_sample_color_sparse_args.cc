// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu> and Anna Arpaci-Dusseau <annaad@mit.edu>
#include "graph.h"
#define C 2
#define P 1.0/C

void ColorSample(Graph &g, float c) {
  std::cout << "|e| before sampling " << g.E() << "\n";
  g.color_sparsify(c);
  std::cout <<  "|e| after sampling " << g.E() << "\n";
}

void TCSolver(Graph &g, uint64_t &total, int, int, vector<float> args) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP 5-Clique Counting (" << num_threads << " threads)\n";
  ColorSample(g,args[0]);
  std::cout << "Sparsified Graph\n";
  Timer t;
  t.Start();
  uint64_t counter = 0;
  #pragma omp parallel for schedule(dynamic, 1) reduction(+:counter)
  for (vidType v1 = 0; v1 < g.V(); v1++) {
    //auto tid = omp_get_thread_num();
    uint64_t local_counter = 0;
    auto y1 = g.N(v1);
    for (auto v2 : y1) {
      auto y1y2 = intersection_set(y1, g.N(v2));
      for (auto v3 : y1y2) {
        auto y1y2y3 = intersection_set(y1y2, g.N(v3));
        for (auto v4 : y1y2y3) {
          local_counter += intersection_num(y1y2y3, g.N(v4));
        }
      }
    }
    counter += local_counter;
  }
  total = counter * (args[0]*args[0]*args[0]*args[0]); // all vertices have same color
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}


