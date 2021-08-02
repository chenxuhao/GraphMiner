#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for(vidType v0 = 0; v0 < g.V(); v0++) {
  auto tid = omp_get_thread_num();
  auto &cmap = cmaps[tid];
  auto y0 = g.N(v0);
  for (auto u : y0) cmap[u] = 1;
  for (auto v1 : y0) {
    if (v1 >= v0) break;
    auto y1 = g.N(v1);
    for (auto v2 : y1) {
      if (cmap[v2] != 1) continue;
      for (auto v3 : y1) {
        if (v3 == v0 || v3 == v2) continue;
        for (auto v4 : g.N(v3)) {
          if (v4 == v1 || v4 == v2) continue;
          if (cmap[v4] == 1) counter ++;
        }
      }
    }
  }
  for (auto u : y0) cmap[u] = 0;
}
