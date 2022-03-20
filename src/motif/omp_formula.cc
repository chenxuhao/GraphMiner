// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>
#include "pattern.hh"
#include "emb_list.h"
#include "cmap_formula.h"
#include "automine_formula.h"

void MotifSolver(Graph &g, int k, std::vector<uint64_t> &total, int , int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Motif solver (" << num_threads << " threads) ...\n";

  int num_patterns = num_possible_patterns[k];
#ifdef USE_CMAP
  //std::vector<uint64_t> counts(num_patterns, 0);
  std::vector<EmbList> emb_lists(num_threads);
  std::vector<std::vector<uint8_t>> ccodes(num_threads);
  auto max_degree = g.get_max_degree();
  for (int tid = 0; tid < num_threads; tid ++) {
    auto &local_ccodes = ccodes[tid];
    local_ccodes.resize(g.size()); // the connectivity code
    std::fill(local_ccodes.begin(), local_ccodes.end(), 0);
    auto &emb_list = emb_lists[tid];
    emb_list.init(k, max_degree, num_patterns);
  }
#endif

  Timer t;
  t.Start();
#ifdef USE_CMAP
  //kmotif(g, k, total, emb_lists, ccodes);
  ccode_kmotif(g, k, total, ccodes);
#else
  automine_kmotif(g, k, total);
#endif
  if (k == 3) {
    total[0] = total[0]/2 - 3 * total[1];
  } else if (k == 4) {
    total[4] = total[4] / 2 - total[5] * 6;
    total[2] = total[2] / 2 - total[4] * 2;
    total[1] = total[1] - total[3] * 4;
    total[0] = total[0] / 6 - total[2] / 3;
  } else {
    exit(1);
  }
  t.Stop();
  std::cout << "runtime [omp_fomula] = " << t.Seconds() << "\n";
}

