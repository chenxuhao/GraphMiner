// This is the implementation using the connectivity map (c-map)
std::cout << "Running the c-map implementation\n";
#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
#if 1
  auto tid = omp_get_thread_num();
  auto &cmap = cmaps[tid];
  for (auto u : g.N(v0)) cmap[u] = 1;
  for (auto v1 : g.N(v0)) {
    if (v1 >= v0) break;
    //uint64_t n = 0;
    VertexSet y0y1;
    for (auto u : g.N(v1)) {
#if 0
      auto c1 = read_cycle();
      auto ccode = cmap[u];
      auto c2 = read_cycle();
      if (nqueries[tid] < NUM_SAMPLES) {
        auto tick = c2 - c1;
        //std::cout << tick << "\n";
        if (tick < 500) {
          nticks[tid] += tick;
          nqueries[tid] ++;
        }
      }
      if (ccode == 1) y0y1.add(u);
#else
      if (cmap[u] == 1) y0y1.add(u);
#endif
    }
    for (auto v2 : y0y1) {
      for (auto v3 : y0y1) {
        if (v3 >= v2) break;
        counter ++;
      }
    }
    //counter += n * (n-1) / 2;
  }
  for (auto u : g.N(v0)) cmap[u] = 0;
#else
  for (auto v1 : g.N(v0)) {
    if (v1 >= v0) break;
    uint64_t n = intersect(g, v0, v1);
    counter += n * (n-1) / 2;
  }
#endif
}
