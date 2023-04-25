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
  std::cout << "OpenMP 9-Clique Counting (" << num_threads << " threads)\n";
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
            auto y1y2y3y4 = intersection_set(y1y2y3, g.N(v4));
            for (auto v5 : y1y2y3y4) {
              auto y1y2y3y4y5 = intersection_set(y1y2y3y4, g.N(v5));
              for (auto v6 : y1y2y3y4y5) {
                auto y1y2y3y4y5y6 = intersection_set(y1y2y3y4y5, g.N(v6));
                for (auto v7 : y1y2y3y4y5y6) {
                  auto y1y2y3y4y5y6y7 = intersection_set(y1y2y3y4y5y6, g.N(v7));
                  for (auto v8 : y1y2y3y4y5y6y7) {
                    local_counter += intersection_num(y1y2y3y4y5y6y7, g.N(v8));
                  }
                }
              }
            }
        }
      }
    }
    counter += local_counter;
  }

  total = counter * ((args[0]*args[0]*args[0]*args[0]*args[0]*args[0]*args[0]*args[0])); // all vertices have same color
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}


