std::vector<std::vector<std::array<vidType, 3>>> all_wedges(num_threads);

#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  int tid = omp_get_thread_num();
  auto &wedges = all_wedges[tid];
  wedges.clear();
  for (vidType v1 : g.N(v0)) {
    if (v1 >= v0) break;
    auto y1 = g.N(v1);
    for (vidType v2 : g.N(v1)) {
      if (v2 >= v0) break;
      wedges.push_back({v0, v1, v2});
    }
  }
  if (!wedges.empty()) {
    // group by v2; for same v2, sort v1 by descending order
    std::sort(wedges.begin(), wedges.end(),
              [](const std::array<vidType, 3> &a,
                 const std::array<vidType, 3> &b) {
                return a[2] < b[2] || (a[2] == b[2] && a[1] > b[1]);
              });

    size_t offset = 0;
    while (offset < wedges.size()) {
      auto v2 = wedges[offset][2];
      size_t next_offset = offset + 1;
      while (next_offset < wedges.size() && wedges[next_offset][2] == v2)
          next_offset++;
      // output
      for (; offset < next_offset; ++offset)
        for (size_t i = offset+1; i < next_offset; ++i)
            ++counter;
    }
  }
}
