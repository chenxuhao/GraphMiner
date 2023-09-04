#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  auto a0 = g.N(v0);
  for (auto v1 : a0) {
    if (v1 > v0) break;
    // (v0, v1) is the chord
    int64_t tri_count = 0;
    int64_t cycle_count = 0;
    auto a1 = g.N(v1);
    auto tri_set = intersection_set(a0, a1); // triangles incident to the chord
    tri_count = tri_set.size();
    for (auto v2 : a0) {
      if (v2 == v1) continue;
      cycle_count += intersection_num(a1, g.N(v2)) - 1; // cycles incident to the chord
    }
    int64_t overlap = 0;
    for (auto v2 : tri_set) {
      overlap += intersection_num(g.N(v2), a0) - 1;
      overlap += intersection_num(g.N(v2), a1) - 1;
    }
    counter += tri_count * cycle_count - overlap;
  }
}
