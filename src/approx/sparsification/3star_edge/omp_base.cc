// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include "graph.h"
#include "pattern.hh"

int nChoosek( unsigned n, unsigned k )
{
    if (k > n) return 0;
    if (k * 2 > n) k = n-k;
    if (k == 0) return 1;

    int result = n;
    for( int i = 2; i <= k; ++i ) {
        result *= (n-i+1);
        result /= i;
    }
    return result;
}

void automine_3star(Graph &g, std::vector<std::vector<uint64_t>> &global_counters) {
  #pragma omp parallel
  {
    auto &counter = global_counters.at(omp_get_thread_num());
    #pragma omp for schedule(dynamic,1)
    for(vidType v0 = 0; v0 < g.V(); v0++) {
      VertexSet y0 = g.N(v0);
      counter[0] += nChoosek(y0.size(),3);
    }
}
}

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
  automine_3star(g, global_counters);
  for (int tid = 0; tid < num_threads; tid++)
    for (int pid = 0; pid < num_patterns; pid++)
      total[pid] += global_counters[tid][pid]; //todo add multiplication factor


  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << "\n";
}

