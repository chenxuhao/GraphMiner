#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  auto y0 = g.N(v0);
  for (vidType v1 : g.N(v0)) {
    if (v1 >= v0) break;
    auto y0y1 = intersection_set(y0, g.N(v1));
    for (vidType v2 : y0y1) {
      for (vidType v3 : y0y1) {
        if (v3 >= v2) break;
        counter += 1;
      }
    }
  }
}
