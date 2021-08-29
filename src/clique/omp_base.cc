// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include "clique.h"
#include "clique_omp.h"

void CliqueSolver(Graph &g, int k, uint64_t &total, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP " << k << "-clique listing (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  double start_time = omp_get_wtime();
  kclique(g, k, total);
  double run_time = omp_get_wtime() - start_time;
  t.Stop();
  std::cout << "runtime [omp_base] = " << run_time << " sec\n";
  return;
}

