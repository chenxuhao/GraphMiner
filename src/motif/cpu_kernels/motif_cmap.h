
void base_3motif(Graph &g, std::vector<std::vector<uint64_t>> &global_counters, 
                  std::vector<std::vector<uint8_t>> &ccodes) {
  #pragma omp parallel for schedule(dynamic,1)
  for (vidType v0 = 0; v0 < g.V(); v0++) {
    auto tid = omp_get_thread_num();
    auto &counter = global_counters.at(tid);
    //auto &local_ccodes = ccodes[tid];
    //update_ccodes(0, g, v0, local_ccodes);
    VertexSet y0 = g.N(v0);
    for (auto v1 : y0) {
      VertexSet y1 = g.N(v1);
      counter[0] += difference_num(y0, y1, v1);
      if (v1 < v0) {
        counter[1] += intersection_num(y0, y1, v1);
      }
    }
  }
}

void ccode_3motif(Graph &g, std::vector<std::vector<uint64_t>> &global_counters, 
                  std::vector<std::vector<uint8_t>> &ccodes) {
  #pragma omp parallel for schedule(dynamic,1)
  for (vidType v0 = 0; v0 < g.V(); v0++) {
    auto tid = omp_get_thread_num();
    auto &counter = global_counters.at(tid);
    auto &local_ccodes = ccodes[tid];
    VertexSet y0 = g.N(v0);
    uint64_t local_counter_0 = 0;
    uint64_t local_counter_1 = 0;
    for (auto u : g.N(v0)) {
      if (u > v0) break;
      local_ccodes[u] = 1;
    }
    for (auto v1 : y0) {
      VertexSet y1 = g.N(v1);
      for (auto v2 : y1) {
        if (v2 >= v0) break;
        if (local_ccodes[v2] == 0) local_counter_0 ++;
      }
      if (v1 < v0) {
        for (auto v2 : y1) {
          if (v2 > v1) break;
          if (local_ccodes[v2] == 1) local_counter_1 ++;
        }
      }
    }
    for (auto u : g.N(v0)) {
      if (u > v0) break;
      local_ccodes[u] = 0;
    }
    counter[0] += local_counter_0;
    counter[1] += local_counter_1;
  }
}

void base_4motif(Graph &g, std::vector<std::vector<uint64_t>> &global_counters, 
                  std::vector<std::vector<uint8_t>> &ccodes) {
  std::cout << "Ad-hoc 4-motif counting\n";
  #pragma omp parallel
  {
    auto &counter = global_counters.at(omp_get_thread_num());
    #pragma omp for schedule(dynamic,1)
    for (vidType v0 = 0; v0 < g.V(); v0++) {
      VertexSet y0 = g.N(v0);
      for (auto v1 : y0) {
        VertexSet y1 = g.N(v1);
        VertexSet y0n1f1 = difference_set(y0, y1, v1);
        for(vidType idx2 = 0; idx2 < y0n1f1.size(); idx2++) {
          vidType v2 = y0n1f1.begin()[idx2];
          VertexSet y2 = g.N(v2);
          counter[0] += difference_num(y0n1f1, y2, v2); // 3-star
        }
      }
      for (auto v1 : y0) {
        if (v1 > v0) break;
        VertexSet y1 = g.N(v1);
        VertexSet y0y1 = intersection_set(y0, y1);
        for (auto v2 : y0y1) {
          VertexSet y2 = g.N(v2);
          counter[4] += difference_num(y0y1, y2, v2); // diamond
          VertexSet n0n1y2; difference_set(n0n1y2, y2, y0); 
          counter[2] += difference_num(n0n1y2, y1); // tailed-triangle
          if (v2 > v1) continue;
          counter[5] += intersection_num(y0y1, y2, v2); // 4-clique
        }
        VertexSet n0y1; difference_set(n0y1, y1, y0);
        for (auto v2 : difference_set(y0, y1)) {
          VertexSet y2 = g.N(v2);
          counter[1] += difference_num(n0y1, y2); // 4-path
        }
        //VertexSet n0f0y1; difference_set(n0f0y1, y1, y0);
        for (auto v2 : difference_set(y0, y1, v1)) {
          VertexSet y2 = g.N(v2);
          counter[3] += intersection_num(n0y1, y2, v0); // 4-cycle
        }
      }
    }
  }
}

void ccode_4motif(Graph &g, std::vector<std::vector<uint64_t>> &global_counters, 
                  std::vector<std::vector<uint8_t>> &ccodes) {
  std::cout << "Ad-hoc 4-motif counting\n";
  #pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    auto &counter = global_counters.at(tid);
    auto &local_ccodes = ccodes[tid];
    #pragma omp for schedule(dynamic,1)
    for(vidType v0 = 0; v0 < g.V(); v0++) {
      VertexSet y0 = g.N(v0);
      update_ccodes(0, g, v0, local_ccodes);
      for (auto v1 : y0) {
        VertexSet y1 = g.N(v1);
        VertexSet y0n1f1 = difference_set(y0, y1, v1);
        for(vidType idx2 = 0; idx2 < y0n1f1.size(); idx2++) {
          vidType v2 = y0n1f1.begin()[idx2];
          VertexSet y2 = g.N(v2);
          counter[0] += difference_num(y0n1f1, y2, v2); // 3-star
        }
      }
      for (auto v1 : y0) {
        if (v1 > v0) break;
        update_ccodes(1, g, v1, local_ccodes);
        VertexSet y1 = g.N(v1);
        VertexSet y0y1 = intersection_set(y0, y1);
        for (auto v2 : y0y1) {
          update_ccodes(2, g, v2, local_ccodes, v2);
          for (auto v3 : y0y1) {
            if (v3 < v2 && local_ccodes[v3] == 3) counter[4] ++; // diamond
          }
          VertexSet y2 = g.N(v2);
          for (auto v3 : y2) {
            if((local_ccodes[v3] & 3) == 0 && v3 != v0 && v3 != v1)
              counter[2] ++; // tailed-triangle
          }
          if (v2 < v1) {
            for (auto v3 : y0y1) {
              if (v3 > v2) break;
              if (local_ccodes[v3] == 7) counter[5] ++; // 4-clique
            }
          }
          resume_ccodes(2, g, v2, local_ccodes, v2);
        }
        VertexSet n0y1; difference_set(n0y1, y1, y0);
        for (auto v2 : y0) {
          if (local_ccodes[v2] == 1) {
            VertexSet y2 = g.N(v2);
            counter[1] += difference_num(n0y1, y2); // 4-path
          }
        }
        //VertexSet n0f0y1; difference_set(n0f0y1, y1, y0);
        //for (auto v2 : difference_set(y0, y1, v1)) {
        for (auto v2 : y0) {
          if (v2 < v1 && local_ccodes[v2] == 1) {
            VertexSet y2 = g.N(v2);
            counter[3] += intersection_num(n0y1, y2, v0); // 4-cycle
          }
        }
        resume_ccodes(1, g, v1, local_ccodes);
      }
      resume_ccodes(0, g, v0, local_ccodes);
    }
  }
}

void ccode_kmotif(Graph &g, unsigned k, std::vector<std::vector<uint64_t>> &counters,
                  std::vector<std::vector<uint8_t>> &ccodes) {
#ifdef USE_CMAP
  if (k == 3) ccode_3motif(g, counters, ccodes);
  else if (k == 4) ccode_4motif(g, counters, ccodes);
#else
  if (k == 3) base_3motif(g, counters, ccodes);
  else if (k == 4) base_4motif(g, counters, ccodes);
#endif
  else return;
}

