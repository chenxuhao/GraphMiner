#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  int64_t count = 0;
  for (auto v1 : g.N(v0)) {
    if (v1 >= v0) break;
    auto a0a1 = intersection_set(g.N(v0), g.N(v1));
    for (auto v2 : a0a1) {
      // for each (v0, v1, v2)
      int64_t tri02 = intersection_num(g.N(v0), g.N(v2));
      int64_t tri12 = intersection_num(g.N(v1), g.N(v2));
      if (tri02 > 1) count += (tri02 - 1) * (tri12 - 1);
      int64_t clique_count = intersection_num(g.N(v2), a0a1);
      count -= clique_count;
    }
  }
  counter += count;
}
