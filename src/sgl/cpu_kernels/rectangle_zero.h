// This is AutoMine/GraphZero implementation
#pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
for (vidType v0 = 0; v0 < g.V(); v0++) {
  VertexSet y0 = g.N(v0); 
  VertexSet y0f0 = bounded(y0,v0);
  for(vidType idx1 = 0; idx1 < y0f0.size(); idx1++) {
    vidType v1 = y0f0.begin()[idx1];
    VertexSet y1 = g.N(v1);
    VertexSet n0f0y1; difference_set(n0f0y1,y1, y0);
    VertexSet y0f0n1f1 = difference_set(y0f0, y1, v1);
    for (vidType idx2 = 0; idx2 < y0f0n1f1.size(); idx2++) {
      vidType v2 = y0f0n1f1.begin()[idx2];
      VertexSet y2 = g.N(v2);
      counter += intersection_num(n0f0y1, y2, v0);
    }
  }
}
