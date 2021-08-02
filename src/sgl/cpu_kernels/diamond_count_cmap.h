// This is the implementation for subgraph counting, not listing
std::cout << "Running subgraph counting using c-map\n";
#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  auto tid = omp_get_thread_num();
  auto &cmap = cmaps[tid];
  for (auto u : g.N(v0)) cmap[u] = 1;
  for (auto v1 : g.N(v0)) {
    if (v1 >= v0) break;
    uint64_t n = 0;
    for (auto u : g.N(v1)) {
      if (cmap[u] == 1) n ++;
    }
    counter += n * (n-1) / 2;
  }
  for (auto u : g.N(v0)) cmap[u] = 0;
}
