// 3-color triangle
#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  auto l0 = g.get_vlabel(v0);
  if (l0 != p.get_vlabel(0)) continue;
  auto y0 = g.N(v0);
  for (vidType v1 : y0) {
    auto l1 = g.get_vlabel(v1);
    if (l1 != p.get_vlabel(1)) continue;
    counter += g.intersect_num(v0, v1, p.get_vlabel(2));
  }
}

