#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  for (vidType v1 : g.N(v0)) {
    for (vidType v2 : g.N(v1)) {
      if (v2 == v0) continue;
      for (vidType v3 : g.N(v2)) {
        if (v3 >= v0) break;
        if (v3 == v1) continue;
        counter += 1;
      }
    }
  }
}
