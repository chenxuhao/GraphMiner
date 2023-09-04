typedef std::array<vidType, 2> Tuple;
std::vector<std::vector<Tuple>> all_wedges(num_threads);
//typedef std::unordered_map<vidType,int64_t> CountMap;
//typedef std::map<vidType,int64_t> CountMap;
//std::vector<CountMap> all_wedges(num_threads);

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
      wedges.push_back({v1, v2});
      //CountMap::iterator it = wedges.find(v2);
      //if (it != wedges.end()) wedges[v2] += 1;
      //else wedges[v2] = 1;
    }
  }
  //for (CountMap::iterator it = wedges.begin(); it != wedges.end(); ++it) {
  //  auto n = it->second;
  //  counter += n * (n-1) / 2;
  //}
  ///*
  if (!wedges.empty()) {
    // group by v2; for same v2, sort v1 by descending order
    std::sort(wedges.begin(), wedges.end(),
              [](const Tuple &a, const Tuple &b) { return a[1] < b[1]; }
    );

    size_t offset = 0;
    while (offset < wedges.size()) {
      auto v2 = wedges[offset][1];
      size_t next_offset = offset + 1;
      while (next_offset < wedges.size() && wedges[next_offset][1] == v2)
          next_offset++;
      auto n = next_offset - offset;
      counter += n * (n-1) / 2;
      offset = next_offset;
    }
  }
  //*/
}
