#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  auto adj0 = g.N(v0);
  for (auto v1 : g.N(v0)) {
    if (v1 >= v0) break;
    int64_t left_tri = 0;
    for (auto v2 : g.N(v0)) {
      if (v2 == v1) continue;
      left_tri += intersection_num_bound_except(adj0, g.N(v2), v2, v1);
    }
    int64_t right_tri = 0;
    auto adj1 = g.N(v1);
    for (auto v4 : g.N(v1)) {
      if (v4 == v0) continue;
      right_tri += intersection_num_bound_except(adj1, g.N(v4), v4, v0);
    }
    int64_t num_prod = left_tri * right_tri;

    // shrinkage
    int64_t num_shrink = 0;
    //auto adj0x1 = intersection_set(adj0, adj1, v1);
    auto adj0x1 = adj0 & adj1;
    // case 1: v2 = v4 & v3 == v5, i.e., 4-clique
    for (auto v2 : adj0x1) {
      //if (v2 >= v1) break;
      num_shrink += intersection_num(adj0x1, g.N(v2), v2);
    }
    // case 2: v2 == v4 & v2 != v5 & v3 != v4 & v3 != v5
    // case 3: v2 == v5 & v2 != v4 & v3 != v4 & v3 != v5
    // case 4: v2 != v4 & v2 != v5 & v3 == v4 & v3 != v5
    // case 5: v2 != v4 & v2 != v5 & v3 != v4 & v3 == v5
    for (auto v2 : adj0x1) {
      auto left_set = intersection_set_except(adj0, g.N(v2), v1);
      int64_t left = left_set.size();
      int64_t right = intersection_num_except(adj1, g.N(v2), v0);
      auto num = left * right;
      num_shrink += num - intersection_num(adj1, left_set);
    }
    assert(num_prod >= num_shrink);
    counter += num_prod - num_shrink;
  }
}
