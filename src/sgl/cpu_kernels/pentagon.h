// This is the baseline implementation for pentagon
#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for(vidType v0 = 0; v0 < g.V(); v0++) {
  for (auto v1 : g.N(v0)) {
    if (v1 >= v0) break;
    auto y1 = g.N(v1);
    for (auto v2 : g.N(v0)) {
      if (v2 >= v1) break;
      for (auto v3 : g.N(v2)) {
        if (v3 >= v0) break;
        if (v3 == v1) continue;
        auto y3 = g.N(v3);
        counter += intersection_num_bound_except(y1, y3, v0, v2);
      }
    }
  }
}

