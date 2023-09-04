#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for(vidType v0 = 0; v0 < g.V(); v0++) {
  for (auto v1 : g.N(v0)) {
    if (v1 >= v0) break;
    auto a0a1 = intersection_set(g.N(v0), g.N(v1));
    for (auto v2 : a0a1) {
      auto a0a2 = intersection_set(g.N(v0), g.N(v2));
      for (auto v3 : a0a2) {
        if (v3 == v1) continue;
        counter += intersection_num_except(g.N(v1), g.N(v2), v0, v3);
      }
    }
  }
}
