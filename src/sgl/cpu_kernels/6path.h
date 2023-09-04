#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  int64_t count = 0;
  for (vidType v1 : g.N(v0)) {
    if (v1 >= v0) break;
    for (vidType v2 : g.N(v0)) {
      if (v2 == v1) continue;
      for (vidType v3 : g.N(v1)) {
        if (v3 == v0 || v3 == v2) continue;
        for (vidType v4 : g.N(v2)) {
          if (v4 == v0 || v4 == v1 || v4 == v3) continue;
          for (vidType v5 : g.N(v3)) {
            if (v5 == v0 || v5 == v1 || v5 == v2 || v5 == v4) continue;
            count += 1;
          }
        }
      }
    }
  }
  counter += count;
}
