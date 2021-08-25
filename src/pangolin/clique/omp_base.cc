// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>

#include <omp.h>
#include "kcl.h"
#include "timer.h"
#define USE_SIMPLE
#define USE_BASE_TYPES
#include "pangolin_cpu/vertex_miner.h"

void KclSolver(Graph &g, unsigned k, uint64_t &total) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP " << k << "-clique listing (" << num_threads << " threads)\n";
  VertexMiner miner(&g, k, num_threads);
  EmbeddingList emb_list;
  emb_list.init(g, k, true);
  uint64_t num = 0;
  unsigned level = 1;
  Timer t;
  t.Start();
  while (1) {
    emb_list.printout_embeddings(level);
    miner.extend_vertex(level, emb_list, num);
    if (level == k-2) break; 
    level ++;
  }
  total = num;
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
}

