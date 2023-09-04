#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  for (vidType v1 : g.N(v0)) {
    for (vidType v2 : g.N(v0)) {
      if (v2 >= v1) break;
      for (vidType v3 : g.N(v0)) {
        if (v3 >= v2) break;
        counter += 1;
      }
    }
  }
}
