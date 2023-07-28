// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu> and Anna Arpaci-Dusseau <annaad@mit.edu>
#include "graph.h"

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
  std::cout << "OpenMP Triangle Counting (" << num_threads << " threads)\n";
  ColorSample(g, args[0]);
  std::cout << "Sparsified Graph\n";
  Timer t;
  t.Start();
  uint64_t counter = 0;
  #pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for(vidType v0 = 0; v0 < g.V(); v0++) {
  for (auto v1 : g.N(v0)) {
    if (v1 >= v0) break;
    auto y1 = g.N(v1);
    for (auto v2 : g.N(v0)) {
      if (v2 >= v1) break;
      for (auto v3 : g.N(v2)) {
        if (v3 >= v0) break;
        if (v3 == v1) continue;
        auto y3 = g.N(v3);
        counter += intersection_num_bound_except(y1, y3, v0, v2);
      }
    }
  }
}
  total = counter * ((args[0]*args[0]*args[0]*args[0]));
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}


