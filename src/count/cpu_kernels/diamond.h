#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  for (auto v1 : g.N(v0)) {
    if (v1 >= v0) break;
    uint64_t n = intersection_num(g.N(v0), g.N(v1));
    if (n > 1) counter += n * (n-1) / 2;
  }
}
