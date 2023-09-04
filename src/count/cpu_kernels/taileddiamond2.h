#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  int64_t deg = g.get_degree(v0);
  for (auto v1 : g.N(v0)) {
    int64_t diamond_count = 0;
    int64_t n = intersection_num(g.N(v0), g.N(v1));
    if (n > 1) diamond_count += n * (n-1) / 2;
    counter += (deg - 3) * diamond_count;
  }
}
