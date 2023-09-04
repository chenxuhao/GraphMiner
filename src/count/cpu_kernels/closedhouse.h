#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  int64_t count = 0;
  for (auto v1 : g.N(v0)) {
    // for each (v0, v1)
    if (v1 >= v0) break;
    auto a0a1 = intersection_set(g.N(v0), g.N(v1));
    int64_t tri_count = a0a1.size();
    int64_t clique_count = 0;
    for (auto v2 : a0a1)
      clique_count += intersection_num(g.N(v2), a0a1);
    if (tri_count > 2) count += clique_count * (tri_count - 2);
  }
  counter += count;
}
