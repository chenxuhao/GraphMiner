#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for(vidType v0 = 0; v0 < g.V(); v0++) {
  auto y0 = g.N(v0);
  for (auto v1 : y0) {
    if (v1 >= v0) break;
    auto y1 = g.N(v1);
    auto y0y1 = intersection_set(y0, y1);
    for (auto v2 : y0y1) {
      for (auto v3 : y1) {
        if (v3 == v0 || v3 == v2) continue;
        auto y3 = g.N(v3);
        counter += intersection_num_except(y0, y3, v1, v2);
      }
    }
  }
}
