// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>

#include <omp.h>
#include "motif.h"
#include "timer.h"
#define USE_PID
#define USE_WEDGE
#define USE_SIMPLE
#define VERTEX_INDUCED
#include "pangolin_cpu/vertex_miner.h"

void MotifSolver(Graph &g, unsigned k, std::vector<uint64_t> &counts) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  VertexMiner miner(&g, k, num_threads);
  EmbeddingList emb_list;
  emb_list.init(g, k);
  int num_patterns = counts.size();
  std::vector<uint64_t> counters(num_patterns, 0);
  std::vector<std::vector<uint64_t>> private_counters(num_threads);
  for (int i = 0; i < num_threads; i++) {
    private_counters[i].resize(num_patterns);
    for (int j = 0; j < num_patterns; j++)
      private_counters[i][j] = 0;
  }
  std::cout << "OpenMP " << k << "-Motif (" << num_threads << " threads) ...\n";

  Timer t;
  t.Start();
  unsigned level = 1;
  while (level < k-2) {
    //emb_list.printout_embeddings(level);
    miner.extend_vertex(level, emb_list);
    level ++;
  }
  if (k < 5) {
    miner.aggregate(level, emb_list, private_counters);
  } else std::cout << "Not supported\n";
  for (int tid = 0; tid < num_threads; tid++)
    for (int pid = 0; pid < num_patterns; pid++)
      counters[pid] += private_counters[tid][pid];
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  for (int i = 0; i < num_patterns; i++) counts[i] = counters[i];
}

