// This is the AutoMine/GraphZero implementation
#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  auto y0 = g.N(v0);
  auto y0f0 = bounded(y0,v0);
  for (vidType idx1 = 0; idx1 < y0f0.size(); idx1++) {
    vidType v1 = y0f0.begin()[idx1];
    auto y1 = g.N(v1);
    auto y0y1 = intersection_set(y0, y1);
    for(vidType idx2 = 0; idx2 < y0y1.size(); idx2++) {
      vidType v2 = y0y1.begin()[idx2];
      auto y2 = g.N(v2);
      counter += difference_num(y0y1, y2, v2);
    }
  }
}
