#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  auto n = g.get_degree(v0);
  if (n > 2) counter += (n-2)*(n-1)*n/6;
}
