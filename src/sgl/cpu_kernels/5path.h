/*
#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  for (vidType v1 : g.N(v0)) {
    for (vidType v2 : g.N(v1)) {
      if (v2 == v0) continue;
      for (vidType v3 : g.N(v2)) {
        if (v3 == v0 || v3 == v1) continue;
        for (vidType v4 : g.N(v3)) {
          if (v4 >= v0) break;
          if (v4 == v1 || v4 == v2) continue;
          counter += 1;
        }
      }
    }
  }
}
*/
#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  for (vidType v1 : g.N(v0)) {
    int64_t count = 0;
    for (vidType v2 : g.N(v0)) {
      if (v2 >= v1) break;
      for (vidType v3 : g.N(v2)) {
        if (v3 == v0 || v3 == v1) continue;
        for (vidType v4 : g.N(v1)) {
          if (v4 == v0 || v4 == v2 || v4 == v3) continue;
          count += 1;
        }
      }
    }
    //printf("(v0=%d,v1=%d): count = %d\n", v0, v1, count);
    counter += count;
  }
}
