// Copyright 2023 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include "graph.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/opadd_reducer.h>

void cilk_kclique(Graph &g, unsigned k, uint64_t &total);

void CliqueSolver(Graph &g, int k, uint64_t &total, int, int) {
  int num_threads = __cilkrts_get_nworkers();
  std::cout << "Cilk " << k << "-clique listing (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  cilk_kclique(g, k, total);
  t.Stop();
  double run_time = t.Seconds();
  std::cout << "runtime [cilk_base] = " << run_time << " sec\n";
  return;
}

void cilk_4clique(Graph &g, uint64_t &total) {
  cilk::opadd_reducer<uint64_t> counter = 0;
  //#pragma grainsize = 1
  cilk_for (vidType v0 = 0; v0 < g.V(); v0++) {
    auto adj0 = g.N(v0);
    cilk_for (vidType i = 0; i < adj0.size(); i++) {
      auto v1 = adj0[i];
      auto y0y1 = adj0 & g.N(v1);
      //for (auto v2 : y0y1) {
      for (vidType j = 0; j < y0y1.size(); j++) {
      //cilk_for (vidType j = 0; j < y0y1.size(); j++) {
        auto v2 = y0y1[j];
        counter += intersection_num(y0y1, g.N(v2));
      }
    }
  }
  total = counter;
}

void cilk_5clique(Graph &g, uint64_t &total) {
  cilk::opadd_reducer<uint64_t> counter = 0;
  //#pragma grainsize = 1
  cilk_for (vidType v0 = 0; v0 < g.V(); v0++) {
    auto adj0 = g.N(v0);
    cilk_for (vidType i = 0; i < adj0.size(); i++) {
      auto v1 = adj0[i];
      auto y0y1 = adj0 & g.N(v1);
      cilk_for (vidType j = 0; j < y0y1.size(); j++) {
        auto v2 = y0y1[j];
        auto y0y1y2 = intersection_set(y0y1, g.N(v2));
        for (auto v3 : y0y1y2) {
        //cilk_for (vidType k = 0; k < y0y1y2.size(); k++) {
          //auto v3 = y0y1y2[k];
          counter += intersection_num(y0y1y2, g.N(v3));
        }
      }
    }
  }
  total = counter;
}

void cilk_kclique(Graph &g, unsigned k, uint64_t &total) {
  std::cout << "Running Cilk k-clique solver\n";
  if (k == 4) {
    cilk_4clique(g, total);
  } else if (k == 5) {
    cilk_5clique(g, total);
  } else {
    std::cout << "Not implemented yet\n";
    exit(0);
  }
}

