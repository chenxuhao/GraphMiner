// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include "graph.h"
#include "pattern.hh"
#include "automine_base.h"

void MotifSolver(Graph &g, int k, std::vector<uint64_t> &total, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Motif solver (" << num_threads << " threads) ...\n";

  int num_patterns = num_possible_patterns[k];
  std::vector<std::vector<uint64_t>> global_counters(num_threads);
  for (int tid = 0; tid < num_threads; tid ++) {
    auto &local_counters = global_counters[tid];
    local_counters.resize(num_patterns);
    std::fill(local_counters.begin(), local_counters.end(), 0);
  } 
  Timer t;
  t.Start();
  automine_kmotif(g, k, global_counters);
  for (int tid = 0; tid < num_threads; tid++)
    for (int pid = 0; pid < num_patterns; pid++)
      total[pid] += global_counters[tid][pid];
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << "\n";
}

