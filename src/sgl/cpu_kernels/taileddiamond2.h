#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  int64_t count = 0;
  for (auto v1 : g.N(v0)) {
    auto a0a1 = intersection_set(g.N(v0), g.N(v1));
    if (a0a1.size() > 1) {
      for (auto v2 : a0a1) {
        for (auto v3 : a0a1) {
          if (v3 >= v2) break;
          for (auto v4 : g.N(v0)) {
            if (v4 == v1 || v4 == v2 || v4 == v3) continue;
            count ++;
          }
        }
      }
    }
  }
  counter += count;
}
