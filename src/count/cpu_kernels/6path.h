#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  int64_t d0 = g.get_degree(v0);
  if (d0 < 2) continue;
  for (vidType v1 : g.N(v0)) {
    if (v1 > v0) break;
    int64_t d1 = g.get_degree(v1);
    if (d1 < 2) continue;

    int64_t count = 0;
    // compute # wedges from v0
    int64_t wedge0 = 0;
    for (vidType v2 : g.N(v0)) {
      if (v2 != v1) {
        for (vidType v4 : g.N(v2)) {
          if (v4 == v0 || v4 == v1) continue;
          wedge0 ++; // v4 != v0 && v4 != v1
        }
      }
    }
    // compute # wedges from v1
    int64_t wedge1 = 0;
    for (vidType v3 : g.N(v1)) {
      if (v3 != v0) {
        for (vidType v5 : g.N(v3)) {
          if (v5 == v0 || v5 == v1) continue;
          wedge1 += 1; // v5 != v0 || v5 != v1
        }
      }
    }
    if (wedge0 == 0 || wedge1 == 0) continue;
    // # of 6-paths for each (v0, v1)
    count += wedge0 * wedge1;
    //printf("(v0=%d,v1=%d): wedge_count %ld ", v0, v1, count);

    // Note that v0 != v1, v2 != v0, v2 != v1, v2 != v4, v3 != v0, v3 != v1, v3 != v5

    // shrinkage 0: (one and two) tailed-triangles: v2 == v3
    auto tri_set = intersection_set(g.N(v0), g.N(v1));
    //int64_t tri_count = tri_set.size();
    int64_t tailedtri_count = 0;
    for (vidType v2 : tri_set) {
      auto deg = g.get_degree(v2);
      if (deg > 2) {
        deg -= 2; // rule out v0 and v1
        tailedtri_count += deg; // one tail triangle, i.e., v2 == v3, v4 == v5
        if (deg > 1) tailedtri_count += deg * (deg - 1); // two tail triangle, i.e., v2 == v3, v4 != v5
      }
    }
    count -= tailedtri_count;
    //printf("tailedtri_count %ld  ", tailedtri_count);

    // Now that v2 != v3

    // shrinkage 1: 5-cycle, i.e., v4 == v5, v3 != v4, v5 != v2
    int64_t cycle5_count = 0;
    for (vidType v2 : g.N(v0)) {
      for (vidType v3 : g.N(v1)) {
        if (v2 == v3 || v3 == v0 || v2 == v1) continue;
        cycle5_count += intersection_num_except(g.N(v2), g.N(v3), v0, v1);
      }
    }
    count -= cycle5_count;
    //printf("cycle5_count %ld  ", cycle5_count);

    // Now that v4 != v5

    // shrinkage 2: left tailed 4-cycle
    // v2 == v5, v3 != v4
    int64_t left_tailed4cycle_count = 0;
    int64_t cycle4_count = 0;
    for (vidType v2 : g.N(v0)) {
      if (v2 == v1) continue;
      auto a1a2_count = intersection_num(g.N(v1), g.N(v2));
      if (a1a2_count < 2) continue;
      int64_t v3_count = a1a2_count - 1; // rule out v0
      cycle4_count += v3_count;
      auto deg = g.get_degree(v2);
      if (deg < 3) continue;
      auto v4_count = deg - 2; // rule out v0 and v3
      if (deg >= 3 && g.is_connected(v1, v2)) v4_count --; // rule out v1
      left_tailed4cycle_count += v3_count * v4_count;
    }
    // shrinkage 3: 4-cycle: v2 == v5, v3 == v4
    count -= cycle4_count;
    //printf("cycle4_count %ld  ", cycle4_count);
    count -= left_tailed4cycle_count;
    //printf("left_tailed4cycle_count %ld  ", left_tailed4cycle_count);

    // shrinkage 4: right tailed 4-cycle: v3 == v4, v2 != v5
    int64_t right_tailed4cycle_count = 0;
    for (vidType v3 : g.N(v1)) {
      if (v3 == v0) continue;
      auto a0a3_count = intersection_num(g.N(v0), g.N(v3));
      if (a0a3_count < 2) continue;
      int64_t v2_count = a0a3_count - 1; // rule out v1
      auto deg = g.get_degree(v3);
      if (deg < 3) continue;
      auto v5_count = deg - 2; // rule out v1 and v2
      if (deg >= 3 && g.is_connected(v0, v3)) v5_count --; // rule out v0
      right_tailed4cycle_count += v2_count * v5_count;
    }
    count -= right_tailed4cycle_count;
    //printf("right_tailed4cycle_count %ld  ", right_tailed4cycle_count);
    //printf("final count %ld\n", count);
    counter += count;
  }
}
