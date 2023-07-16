#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  auto a0 = g.N(v0);
  int64_t tri_count = 0;
  int64_t diamond_count = 0;
  for (vidType v1 : a0) {
    int64_t count = intersection_num(a0, g.N(v1), v1); // v2 < v1
    tri_count += count;
    //if (count > 1) diamond_count += count * (count - 1) / 2;
    int64_t count0 = intersection_num(a0, g.N(v1));
    if (count0 > 1) diamond_count += count0 * (count0 - 1) / 2;
    //printf("\t v0=%d, v1=%d, count=%d, count0=%d\n", v0, v1, count, count0);
  }
  //printf("v0=%d, tri_count=%d, diamond_count=%d\n", v0, tri_count, diamond_count);
  if (tri_count > 1) {
    counter += tri_count * (tri_count - 1) / 2 - diamond_count;
  }
}
