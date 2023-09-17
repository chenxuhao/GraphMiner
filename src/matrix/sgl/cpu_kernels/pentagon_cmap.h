// This is the c-map implementation for pentagon
#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  auto tid = omp_get_thread_num();
  auto &cmap = cmaps[tid];
  for (auto v1 : g.N(v0)) {
    for (auto u : g.N(v1)) {
      if (u >= v0) break;
      cmap[u] = 1;
    }
    if (v1 < v0) {
      for (auto v2 : g.N(v0)) {
        if (v2 >= v1) break;
        for (auto v3 : g.N(v2)) {
          if (v3 >= v0) break;
          if (v3 == v1) continue;
          for (auto v4 : g.N(v3)) {
            if (v4 >= v0) break;
            if (v4 == v2) continue;
            if (cmap[v4] == 1)
              counter ++;
          }
        }
      }
    }
    for (auto u : g.N(v1)) {
      if (u >= v0) break;
      cmap[u] = 0;
    }
  }
}
