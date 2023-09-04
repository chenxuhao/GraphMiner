#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  for (vidType v1 : g.N(v0)) {
    auto y0y1 = intersection_set(g.N(v0), g.N(v1), v1);
    for (vidType v2 : y0y1) {
      for (vidType v3 : g.N(v0)) {
        if (v3 == v1 || v3 == v2) continue;
        counter += 1;
      }
    }
  }
}
