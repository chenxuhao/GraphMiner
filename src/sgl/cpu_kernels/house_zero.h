// This is AutoMine/GraphZero implementation
#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for(vidType v0 = 0; v0 < g.V(); v0++) {
  auto y0 = g.N(v0);
  auto y0f0 = bounded(y0,v0);
  for(vidType idx1 = 0; idx1 < y0f0.size(); idx1++) {
    auto v1 = y0f0.begin()[idx1];
    auto y1 = g.N(v1);
    auto y0y1 = intersection_set(y0, y1);
    VertexSet n0y1; difference_set(n0y1,y1, y0);
    auto y0n1 = difference_set(y0, y1);
    for(vidType idx2 = 0; idx2 < y0y1.size(); idx2++) {
      auto v2 = y0y1.begin()[idx2];
      auto y2 = g.N(v2);
      auto n0y1n2 = difference_set(n0y1, y2);
      auto y0n1n2 = difference_set(y0n1, y2);
      for(vidType idx3 = 0; idx3 < n0y1n2.size(); idx3++) {
        auto v3 = n0y1n2.begin()[idx3];
        auto y3 = g.N(v3);
        counter += intersection_num(y0n1n2, y3);
      }
    }
  }
}
