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
    if(v0 != 0 && numIters % v0 == 0) { printf("percent done: %d \n", numIters / v0); } 
    uint64_t local_counter = 0;
    //auto tid = omp_get_thread_num();
    auto y0 = g.N(v0);
    int iter = 0;
    for (auto v1 : y0) {
      if(iter > args[1] * y0.size()) break;
      int j1 = 0;
      auto y1 = g.N(v1);
      for (auto v2 : y1) {
          if(j1 > args[2] * y1.size()) break;
          auto y2 = g.N(v2);
          int j2 = 0;
          for (auto v3 : y2) {
              if(j2 > args[3] * y2.size()) break;
              auto y3 = g.N(v3);
              int j3 = 0;
              for (auto v4 : y3) {
                  if(j3 > args[4] * y3.size()) break;
                  float p = args[0] * args[1] * args[2] * args[3];
                  VertexSet vs = VertexSet();
                  vs.add(v4);
                  local_counter += intersection_num(vs, g.N(v0)) / p;
                  j3 += 1;
               }
              j2+=1;
            }
        j1+=1;
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


