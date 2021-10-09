
void base_3motif(Graph &g, std::vector<std::vector<uint64_t>> &global_counters,
                 std::vector<std::vector<uint8_t>> &ccodes) {
  #pragma omp parallel for schedule(dynamic,1) //reduction(+:)
  for (vidType v0 = 0; v0 < g.V(); v0++) {
    auto &counter = global_counters.at(omp_get_thread_num());
    VertexSet y0 = g.N(v0);
    uint64_t n = (uint64_t)y0.size();
    counter[0] += n * (n - 1);
    uint64_t num = 0;
    for (auto v1 : y0) {
      if (v1 > v0) break;
#if 0
      auto it = g.edge_begin(v0);
      for (auto v2 : g.N(v1)) {
        if (v2 > v1) break;
        while (g.getEdgeDst(it) < v2)
          it ++;
        if (v2 == g.getEdgeDst(it))
          num += 1;
      }
#else
      VertexSet y1 = g.N(v1);
      num += (uint64_t)intersection_num(y0, y1, v1);
#endif
    }
    counter[1] += num;
  }
}

void base_4motif(Graph &g, std::vector<std::vector<uint64_t>> &global_counters, 
                 std::vector<std::vector<uint8_t>> &ccodes) {
  std::cout << "Ad-hoc 4-motif counting\n";
  #pragma omp parallel
  {
    auto &counter = global_counters.at(omp_get_thread_num());
    #pragma omp for schedule(dynamic,1) nowait
    for(vidType v0 = 0; v0 < g.V(); v0++) {
      VertexSet y0 = g.N(v0);
      for (auto v1 : y0) {
        if (v1 > v0) break;
        VertexSet y1 = g.N(v1);
        VertexSet y0y1 = intersection_set(y0, y1);
        uint64_t tri = y0y1.size();
        counter[4] += tri * (tri - 1);
        uint64_t staru = y0.size() - tri - 1;
        uint64_t starv = y1.size() - tri - 1;
        counter[2] += tri * (staru + starv);
        counter[1] += staru * starv;
        counter[0] += staru * (staru - 1);
        counter[0] += starv * (starv - 1);
        for (auto v2 : y0y1) {
          if (v2 > v1) break;
          VertexSet y2 = g.N(v2);
          counter[5] += intersection_num(y0y1, y2, v2); // 4-clique
        }
        VertexSet n0y1; difference_set(n0y1, y1, y0);
        VertexSet y0n1f1 = difference_set(y0, y1, v1);
        for (auto v2 : y0n1f1) {
          if (v2 > v1) break;
          VertexSet y2 = g.N(v2);
          counter[3] += intersection_num(n0y1, y2, v0); // 4-cycle
        }
      }
    }
  }
}

void ccode_3motif(Graph &g, std::vector<std::vector<uint64_t>> &global_counters,
                 std::vector<std::vector<uint8_t>> &ccodes) {
  #pragma omp parallel for schedule(dynamic,1) //reduction(+:)
  for(vidType v0 = 0; v0 < g.V(); v0++) {
    auto tid = omp_get_thread_num();
    auto &local_ccodes = ccodes.at(tid);
    auto &counter = global_counters.at(tid);
    update_ccodes(0, g, v0, local_ccodes, v0);
    VertexSet y0 = g.N(v0);
    uint64_t n = (uint64_t)y0.size();
    counter[0] += n * (n - 1);
    uint64_t num = 0;
    for (auto v1 : y0) {
      if (v1 > v0) break;
      //update_ccodes(1, g, v1, local_ccodes);
      for (auto v2 : g.N(v1)) {
        if (v2 > v1) break;
        if (local_ccodes[v2] == 1)
          num += 1;
      }
    }
    resume_ccodes(0, g, v0, local_ccodes, v0);
    counter[1] += num;
  }
}

void ccode_4motif(Graph &g, std::vector<std::vector<uint64_t>> &global_counters, 
                 std::vector<std::vector<uint8_t>> &ccodes) {
  std::cout << "Ad-hoc 4-motif counting\n";
  #pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    auto &counter = global_counters.at(tid);
    auto &local_ccodes = ccodes.at(tid);
    #pragma omp for schedule(dynamic,1) nowait
    for(vidType v0 = 0; v0 < g.V(); v0++) {
      update_ccodes(0, g, v0, local_ccodes, v0);
      VertexSet y0 = g.N(v0);
      for (auto v1 : y0) {
        if (v1 > v0) break;
        update_ccodes(1, g, v1, local_ccodes);
        VertexSet y1 = g.N(v1);
        VertexSet y0y1 = intersection_set(y0, y1);
        uint64_t tri = y0y1.size();
        counter[4] += tri * (tri - 1);
        uint64_t staru = y0.size() - tri - 1;
        uint64_t starv = y1.size() - tri - 1;
        counter[2] += tri * (staru + starv);
        counter[1] += staru * starv;
        counter[0] += staru * (staru - 1);
        counter[0] += starv * (starv - 1);
        for (auto v2 : y0y1) {
          if (v2 > v1) break;
          for (auto v3 : g.N(v2)) {
            if (v3 > v2) break;
            if (local_ccodes[v3] == 3) counter[5] ++; // 4-clique
          }
        }
        for (auto v2 : y0) {
          if (v2 >= v1) break;
          if (local_ccodes[v2] != 1) continue;
          for (auto v3 : g.N(v2)) {
            if (v3 >= v0) break;
            if (local_ccodes[v3] == 2) counter[3] ++; // 4-cycle
          }
        }
        resume_ccodes(1, g, v1, local_ccodes);
      }
      resume_ccodes(0, g, v0, local_ccodes, v0);
    }
  }
}

void ccode_kmotif(Graph &g, unsigned k, std::vector<uint64_t> &counts,
                  std::vector<std::vector<uint8_t>> &ccodes) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  int num_patterns = num_possible_patterns[k];
  //std::cout << "num_threads: " << num_threads << " num_patterns: " << num_patterns << "\n";
  std::vector<std::vector<uint64_t>> global_counters(num_threads);
  for (int tid = 0; tid < num_threads; tid ++) {
    auto &counters = global_counters[tid];
    counters.resize(num_patterns);
    std::fill(counters.begin(), counters.end(), 0);
  }
#ifdef USE_CMAP
  if (k == 3) ccode_3motif(g, global_counters, ccodes);
  else if (k == 4) ccode_4motif(g, global_counters, ccodes);
#else
  if (k == 3) base_3motif(g, global_counters, ccodes);
  else if (k == 4) base_4motif(g, global_counters, ccodes);
#endif
  else { std::cout << "Not implemented yet\n"; exit(0); }
  for (int tid = 0; tid < num_threads; tid++)
    for (int pid = 0; pid < num_patterns; pid++)
      counts[pid] += global_counters[tid][pid];
}

