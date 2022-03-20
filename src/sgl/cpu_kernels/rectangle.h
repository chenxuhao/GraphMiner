#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  for (vidType v1 : g.N(v0)) {
    if (v1 >= v0) break;
    auto y1 = g.N(v1);
    for (vidType v2 : g.N(v0)) {
      if (v2 >= v1) break;
      counter += intersection_num(y1, g.N(v2), v0);
    }
  }
}
