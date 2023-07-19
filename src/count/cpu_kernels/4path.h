#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  int64_t count = 0;
  int64_t d0 = g.get_degree(v0);
  if (d0 < 2) continue;
  for (vidType v1 : g.N(v0)) {
    if (v1 > v0) break;
    int64_t d1 = g.get_degree(v1);
    if (d1 < 2) continue;
    int64_t tri_count  = intersection_num(g.N(v0), g.N(v1));
    //for (vidType v2 : g.N(v1)) {
      //for (vidType v3 : g.N(v2)) {
        //if (v3 >= v0) break;
    count += (d0-1) * (d1-1) - tri_count;
  }
  counter += count;
}
