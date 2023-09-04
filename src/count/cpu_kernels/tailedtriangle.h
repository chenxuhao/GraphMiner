#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  auto deg = g.get_degree(v0);
  if (deg > 2) {
    uint64_t tri_count = 0;
    for (auto v1 : g.N(v0)) {
      uint64_t n = intersection_num(g.N(v0), g.N(v1), v1);
      tri_count += n;
    }
    counter += tri_count * (deg - 2);
  }
}
