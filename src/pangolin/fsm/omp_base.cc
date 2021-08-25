// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include <omp.h>
#include "fsm.h"
#include "timer.h"
#define USE_PID
#define USE_DOMAIN
#define EDGE_INDUCED
#define ENABLE_LABEL
#include "pangolin_cpu/edge_miner.h"
#define FSM_VARIANT "omp_base"

void FsmSolver(Graph &g, unsigned k, unsigned minsup, int npatterns, int &total_num) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  EdgeMiner miner(&g, k, num_threads);
  miner.set_threshold(minsup);
  EmbeddingList emb_list;
  emb_list.init(g, k+1);
  std::cout << "OpenMP " << k << "-FSM (" << num_threads << " threads)\n";

  Timer t;
  t.Start();
  unsigned level = 1;
  int num_freq_patterns = miner.init_aggregator();
  total_num += num_freq_patterns;
  if (num_freq_patterns == 0) {
    std::cout << "No frequent pattern found\n\n";
    return;
  }
  std::cout << "Number of frequent single-edge patterns: " << num_freq_patterns << "\n";
  miner.init_filter(emb_list);
  emb_list.printout_embeddings(level);

  while (1) {
    miner.extend_edge(level, emb_list);
    level ++;
    miner.quick_aggregate(level, emb_list);
    miner.merge_qp_map(level+1);
    miner.canonical_aggregate();
    miner.merge_cg_map(level+1);
    num_freq_patterns = miner.support_count();
    std::cout << "num_frequent_patterns: " << num_freq_patterns << "\n";
    //miner.printout_agg();

    total_num += num_freq_patterns;
    if (num_freq_patterns == 0) break;
    if (level == k) break;
    miner.filter(level, emb_list);
    emb_list.printout_embeddings(level);
  }
  t.Stop();
  std::cout << "\n\tNumber of frequent patterns (minsup=" << minsup << "): " << total_num << "\n\n";
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
}

