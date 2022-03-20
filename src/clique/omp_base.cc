// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include "graph.h"
#include "automine_omp.h"
#include "clique_cmap.h"

void cmap_kclique(Graph &g, int k, uint64_t &total, 
                  std::vector<cmap8_t> &cmaps,
                  std::vector<EmbList> &emb_lists) {
  if (k == 4) {
    cmap_4clique(g, total, cmaps, emb_lists);
  } else if (k == 5) {
    cmap_5clique(g, total, cmaps, emb_lists);
  } else {
    std::cout << "Not implemented yet";
  }
}

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
  automine_kclique(g, k, total);
  //cmap_kclique(g, k, total);
  double run_time = omp_get_wtime() - start_time;
  t.Stop();
  std::cout << "runtime [omp_base] = " << run_time << " sec\n";
  return;
}

