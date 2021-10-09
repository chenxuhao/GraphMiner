// This code is modified from AutoMine and GraphZero
// Daniel Mawhirter and Bo Wu. SOSP 2019.
// AutoMine: Harmonizing High-Level Abstraction and High Performance for Graph Mining

// ad-hoc 3-clique with on-the-fly symmetry breaking
void automine_3clique_sb(Graph &g, uint64_t &total) {
  uint64_t counter = 0;
  #pragma omp parallel for schedule(dynamic, 1) reduction(+:counter)
  for (vidType v0 = 0; v0 < g.V(); v0++) {
    uint64_t local_counter = 0;
    auto y0 = g.N(v0);
    auto y0f0 = bounded(y0,v0);
    for (auto v1 : y0f0)
      local_counter += intersection_num(y0f0, g.N(v1), v1);
    counter += local_counter;
  }
  total = counter;
}

// ad-hoc 3-clique (use DAG)
void automine_3clique(Graph &g, uint64_t &total) {
  uint64_t counter = 0;
  #pragma omp parallel for schedule(dynamic, 1) reduction(+:counter)
  for (vidType v0 = 0; v0 < g.V(); v0++) {
    uint64_t local_counter = 0;
    auto y0 = g.N(v0);
    for (auto v1 : y0)
      local_counter += intersection_num(y0, g.N(v1));
    counter += local_counter;
  }
  total = counter;
}

// ad-hoc 4-clique with on-the-fly symmetry breaking
void automine_4clique_sb(Graph &g, uint64_t &total) {
  uint64_t counter = 0;
  #pragma omp parallel for schedule(dynamic, 1) reduction(+:counter)
  for (vidType v0 = 0; v0 < g.V(); v0++) {
    auto y0 = g.N(v0);
    for (auto v1 : y0) {
      if (v1 >= v0) break;
      auto y0y1 = y0 & g.N(v1);
      for (auto v2 : y0y1) {
        if (v2 >= v1) break;
        counter += intersection_num(y0y1, g.N(v2), v2);
      }
    }
  }
  total = counter;
}

// ad-hoc 4-clique (use DAG)
void automine_4clique(Graph &g, uint64_t &total) {
  uint64_t counter = 0;
  #pragma omp parallel for schedule(dynamic, 1) reduction(+:counter)
  for (vidType v0 = 0; v0 < g.V(); v0++) {
    uint64_t local_counter = 0;
    auto y0 = g.N(v0);
    for (auto v1 : y0) {
      auto y0y1 = y0 & g.N(v1);
      for (auto v2 : y0y1) {
        local_counter += intersection_num(y0y1, g.N(v2));
      }
    }
    counter += local_counter;
  }
  total = counter;
}

// ad-hoc 5-clique with on-the-fly symmetry breaking
void automine_5clique_sb(Graph &g, uint64_t &total) {
  uint64_t counter = 0;
  #pragma omp parallel for schedule(dynamic, 1) reduction(+:counter)
  for (vidType v1 = 0; v1 < g.V(); v1++) {
    uint64_t local_counter = 0;
    auto y1 = g.N(v1);
    for (auto v2 : y1) {
      if (v2 > v1) break;
      auto y1y2 = intersection_set(y1, g.N(v2));
      for (auto v3 : y1y2) {
        if (v3 > v2) break;
        auto y1y2y3 = intersection_set(y1y2, g.N(v3));
        for (auto v4 : y1y2y3) {
          if (v4 > v3) break;
          local_counter += intersection_num(y1y2y3, g.N(v4), v4);
        }
      }
    }
    counter += local_counter;
  }
  total = counter;
}

// ad-hoc 5-clique (use DAG)
void automine_5clique(Graph &g, uint64_t &total) {
  uint64_t counter = 0;
  #pragma omp parallel for schedule(dynamic, 1) reduction(+:counter)
  for (vidType v1 = 0; v1 < g.V(); v1++) {
    uint64_t local_counter = 0;
    auto y1 = g.N(v1);
    for (auto v2 : y1) {
      auto y1y2 = intersection_set(y1, g.N(v2));
      for (auto v3 : y1y2) {
        auto y1y2y3 = intersection_set(y1y2, g.N(v3));
        for (auto v4 : y1y2y3) {
          local_counter += intersection_num(y1y2y3, g.N(v4));
        }
      }
    }
    counter += local_counter;
  }
  total = counter;
}

void kclique(Graph &g, unsigned k, uint64_t &total) {
  std::cout << "Running AutoMine k-clique solver\n";
  if (k == 3) {
#if USE_DAG == 1
    automine_3clique(g, total);
#else
    automine_3clique_sb(g, total);
#endif
  } else if (k == 4) {
#if USE_DAG == 1
    automine_4clique(g, total);
#else
    automine_4clique_sb(g, total);
#endif
  } else if (k == 5) {
#if USE_DAG == 1
    automine_5clique(g, total);
#else
    automine_4clique_sb(g, total);
#endif
  } else {
    std::cout << "Not implemented yet\n";
    exit(0);
  }
}

