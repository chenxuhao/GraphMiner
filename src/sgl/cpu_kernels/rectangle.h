// This code is modified from AutoMine and GraphZero
// Daniel Mawhirter and Bo Wu. SOSP 2019.
// AutoMine: Harmonizing High-Level Abstraction and High Performance for Graph Mining
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
