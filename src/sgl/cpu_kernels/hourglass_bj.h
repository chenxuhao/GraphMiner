#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  auto a0 = g.N(v0);
  int64_t tri_count = 0;
  int64_t diamond_count = 0;
  for (vidType v1 : a0) {
    int64_t count = intersection_num(a0, g.N(v1), v1); // v2 < v1
    tri_count += count;
    int64_t count0 = intersection_num(a0, g.N(v1));
    if (count0 > 1) diamond_count += count0 * (count0 - 1) / 2;
  }
  if (tri_count > 1) {
    counter += tri_count * (tri_count - 1) / 2 - diamond_count;
  }
}
/*
typedef std::array<vidType, 2> tuple_t;
std::vector<std::vector<tuple_t>> all_triangles(num_threads);

#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  int tid = omp_get_thread_num();
  auto &triangles = all_triangles[tid];
  triangles.clear();
  auto adj0 = g.N(v0);
  for (vidType v1 : g.N(v0)) { // v1 \in adj_v0
    auto adj0adj1 = intersection_set(adj0, g.N(v1), v1);
    for (vidType v2 : adj0adj1) { // v2 \in adj_v0 \cap adj_v1, v2 < v1
      triangles.push_back({v1, v2});
    }
  }
  if (!triangles.empty()) {
    // group by v2; for same v2, sort v1 by descending order
    std::sort(triangles.begin(), triangles.end(),
              [](const tuple_t &a, const tuple_t &b) {
                return a[1] < b[1] || (a[1] == b[1] && a[0] > b[0]);
              });
    // todo: enumerate combinations of pairs <v1, v2> and <v3, v4>
    // symmetry breaking: v3 < v1, v3 != v2, v4 < v3
  }
}
*/
