// This code is modified from AutoMine and GraphZero
// Daniel Mawhirter and Bo Wu. SOSP 2019.
// AutoMine: Harmonizing High-Level Abstraction and High Performance for Graph Mining
// Please do not copy or distribute without permission of the author

void automine_3motif(Graph &g, std::vector<std::vector<uint64_t>> &global_counters) {
  #pragma omp parallel
  {
    auto &counter = global_counters.at(omp_get_thread_num());
    #pragma omp for schedule(dynamic,1) nowait
    for(vidType v0 = 0; v0 < g.V(); v0++) {
      VertexSet y0 = g.N(v0);
      uint64_t n = (uint64_t)y0.size();
      counter[0] += n * (n - 1);
      VertexSet y0f0 = bounded(y0,v0);
      for(vidType idx1 = 0; idx1 < y0f0.size(); idx1++) {
        vidType v1 = y0f0.begin()[idx1];
        VertexSet y1 = g.N(v1);
        counter[1] += (uint64_t)intersection_num(y0, y1, v1);
      }
    }
  }
}

void automine_4motif(Graph &g, std::vector<std::vector<uint64_t>> &global_counters) {
  #pragma omp parallel
  {
    auto &counter = global_counters.at(omp_get_thread_num());
    #pragma omp for schedule(dynamic,1) nowait
    for(vidType v0 = 0; v0 < g.V(); v0++) {
      VertexSet y0 = g.N(v0);
      VertexSet y0f0 = bounded(y0,v0);
      for(vidType idx1 = 0; idx1 < y0f0.size(); idx1++) {
        vidType v1 = y0f0.begin()[idx1];
        VertexSet y1 = g.N(v1);
        uint64_t tri = intersection_num(y0, y1);
        counter[4] += tri * (tri - 1);
        uint64_t staru = y0.size() - tri - 1;
        uint64_t starv = y1.size() - tri - 1;
        counter[2] += tri * (staru + starv);
        counter[1] += staru * starv;
        counter[0] += staru * (staru - 1);
        counter[0] += starv * (starv - 1);
        VertexSet y0f0y1f1 = intersection_set(y0f0, y1, v1);
        VertexSet n0f0y1; difference_set(n0f0y1,y1, y0);
        VertexSet y0f0n1f1 = difference_set(y0f0, y1, v1);
        for(vidType idx2 = 0; idx2 < y0f0y1f1.size(); idx2++) {
          vidType v2 = y0f0y1f1.begin()[idx2];
          VertexSet y2 = g.N(v2);
          counter[5] += intersection_num(y0f0y1f1, y2, v2);
        }
        for(vidType idx2 = 0; idx2 < y0f0n1f1.size(); idx2++) {
          vidType v2 = y0f0n1f1.begin()[idx2];
          VertexSet y2 = g.N(v2);
          counter[3] += intersection_num(n0f0y1, y2, v0);
        }
      }
    }
  }
}

void automine_5motif(Graph &g, std::vector<std::vector<uint64_t>> &global_counters) {
}

void automine_kmotif(Graph &g, unsigned k, std::vector<uint64_t> &counts) {
  std::cout << "Running AutoMine k-motif solver using formula\n";
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  int num_patterns = num_possible_patterns[k];
  std::vector<std::vector<uint64_t>> global_counters(num_threads);
  for (int tid = 0; tid < num_threads; tid ++) {
    auto &counters = global_counters[tid];
    counters.resize(num_patterns);
    std::fill(counters.begin(), counters.end(), 0);
  }
  if (k == 3) {
    automine_3motif(g, global_counters);
  } else if (k == 4) {
    automine_4motif(g, global_counters);
  } else if (k == 5) {
    automine_5motif(g, global_counters);
  } else {
    std::cout << "Not implemented yet\n";
    exit(0);
  }
  for (int tid = 0; tid < num_threads; tid++)
    for (int pid = 0; pid < num_patterns; pid++)
      counts[pid] += global_counters[tid][pid];
}

