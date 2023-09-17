// This is AutoMine/GraphZero implementation
#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for(vidType v0 = 0; v0 < g.V(); v0++) {
  auto y0 = g.N(v0);
  auto y0f0 = bounded(y0,v0);
  for(vidType idx1 = 0; idx1 < y0f0.size(); idx1++) {
    auto v1 = y0f0.begin()[idx1];
    auto y1 = g.N(v1);
    VertexSet n0f0y1;
    difference_set(n0f0y1,y1, y0);
    auto y0f0n1f1 = difference_set(y0f0, y1, v1);
    for(vidType idx2 = 0; idx2 < y0f0n1f1.size(); idx2++) {
      auto v2 = y0f0n1f1.begin()[idx2];
      auto y2 = g.N(v2);
      VertexSet n0f0n1y2; 
      difference_set(n0f0n1y2,difference_set(n0f0n1y2,y2, y0), y1);
      auto n0y1n2f0 = difference_set(n0f0y1, y2, v0);
      for(vidType idx3 = 0; idx3 < n0y1n2f0.size(); idx3++) {
        auto v3 = n0y1n2f0.begin()[idx3];
        auto y3 = g.N(v3);
        counter += intersection_num(n0f0n1y2, y3, v0);
      }
    }
  }
}
