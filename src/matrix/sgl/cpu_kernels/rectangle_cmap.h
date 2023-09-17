#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  auto tid = omp_get_thread_num();
  auto &cmap = cmaps.at(tid);
  for (auto v1 : g.N(v0)) {
    for (auto u : g.N(v1)) {
      if (u >= v0) break;
      cmap[u] = 1;
    }
    if (v1 >= v0) break;
    for (auto v2 : g.N(v0)) {
      if (v2 >= v1) break;
      for (auto v3 : g.N(v2)) {
        if (v3 >= v0) break;
        auto c1 = read_cycle();
        if (cmap[v3] == 1) counter ++;
        auto c2 = read_cycle();
#ifdef PROFILE_LATENCY
        if (nqueries[tid] < NUM_SAMPLES) {
          auto tick = c2 - c1;
          if (tick < 500) {
            nticks[tid] += tick;
            nqueries[tid] ++;
          }
        }
#endif
      }
    }
    for (auto u : g.N(v1)) {
      if (u >= v0) break;
      cmap[u] = 0;
    }
  }
}

