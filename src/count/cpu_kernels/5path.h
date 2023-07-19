#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  int64_t d0 = g.get_degree(v0);
  if (d0 < 2) continue;
  for (vidType v1 : g.N(v0)) {
    int64_t count = 0;
    // for each edge (v0, v1)
    int64_t d1 = g.get_degree(v1);
    if (d1 < 2) continue;
    int64_t wedge_count = 0;
    int64_t cycle_count = 0;
    for (vidType v2 : g.N(v0)) {
      if (v2 >= v1) break; // make sure v2 < v1
      int64_t d2 = g.get_degree(v2);
      if (d2 > 1) wedge_count += d2 - 1;
      cycle_count += intersection_num(g.N(v1), g.N(v2)) - 1;
    }
    count += wedge_count * (d1 - 1) - cycle_count;
    auto a0a1 = intersection_set(g.N(v0), g.N(v1), v1);
    int64_t tri_count = a0a1.size();
    int64_t tt1_count = 0; // tailed_triangle
    int64_t tt2_count = 0;
    tt1_count = tri_count * (d1 - 2);
    for (vidType v2 : a0a1) {
      int64_t d2 = g.get_degree(v2);
      if (d2 > 2) tt2_count += (d2 - 2);
    }
    count = count - tt1_count - tt2_count - tri_count;
    //printf("(v0=%d,v1=%d): wedge_count = %d, cycle_count = %d, tri_count=%d, tt1_count = %d, tt2_count = %d, count = %d\n", v0, v1, wedge_count, cycle_count, tri_count, tt1_count, tt2_count, count);
    counter += count;
  }
}
