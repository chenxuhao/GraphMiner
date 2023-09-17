#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  auto adj0 = g.N(v0);
  for (vidType v1 : g.N(v0)) { // v1 \in adj_v0
    auto adj0adj1 = intersection_set(adj0, g.N(v1), v1);
    for (vidType v2 : adj0adj1) { // v2 \in adj_v0 \cap adj_v1, v2 < v1
      for (vidType v3 : adj0) { // v3 \in adj_v0
        if (v3 == v1 || v3 == v2) continue;
        auto adj3 = g.N(v3);
        for (vidType v4 : adj3) { // v4 \in adj_v3, v4 < v1
          if (v4 >= v1) break;
          if (v4 == v0 || v4 == v2) continue;
          counter += intersection_set(adj3, g.N(v4), v4); // v5 \in adj_v3 \cap adj_v4, v5 < v4
        }
      }
    }
  }
}
