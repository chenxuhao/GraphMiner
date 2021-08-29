#include "query.h"

void QuerySolver(Graph &g, Pattern &p, uint64_t &total, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Graph Querying (" << num_threads << " threads)\n";
  uint64_t counter = 0;
  int nlabels = g.get_vertex_classes();
  if (nlabels < 3) return;
  //for (vidType v = 0; v < g.V(); v++)
  //  std::cout << "v" << v << " label: " << unsigned(g.get_vlabel(v)) << "\n";
  vlabel_t C1 = p.label(1), C2 = p.label(2), C3 = p.label(3);
  std::cout << "C1 " << unsigned(C1) << " C2 " << unsigned(C2) << " C3 " << unsigned(C3) << "\n";

  Timer t;
  t.Start();
  // 3-color triangle
  #pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
  for (vidType v0 = 0; v0 < g.V(); v0++) {
    auto l0 = g.get_vlabel(v0);
    if (l0 != C1) continue;
    auto y0 = g.N(v0);
    for (vidType v1 : y0) {
      auto l1 = g.get_vlabel(v1);
      if (l1 != C2) continue;
      counter += g.intersect_num(v0, v1, C3);
    }
  }
  total = counter;
  t.Stop();
  std::cout << "runtime = " <<  t.Seconds() << " seconds\n";
  return;
}

