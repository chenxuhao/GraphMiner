#include "graph.h"
#include "work_stealer.h"

void SglSolver(Graph &g, Pattern &p, uint64_t &total) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("OpenMP edge-induced subgraph listing (%d threads) ...\n", num_threads);
  uint64_t counter = 0;
  std::vector<std::vector<uint8_t>> cmaps(num_threads);
  for (int tid = 0; tid < num_threads; tid ++) {
    auto &cmap = cmaps[tid];
    cmap.resize(g.size()); // the connectivity code
    std::fill(cmap.begin(), cmap.end(), 0);
  }
  LoadBalancer lb(num_threads);

  Timer t;
  t.Start();
  std::cout << "Running the load balancing version\n";
  if (p.is_house()) {
    #include "house_zero.h"
  } else if (p.is_pentagon()) {
    #include "pentagon_zero.h"
  } else if (p.is_rectangle()) {
    #include "rectangle_steal.h"
  } else {
    #include "diamond_zero.h"
  }
  total = counter;
  t.Stop();
  std::cout << "runtime = " <<  t.Seconds() << " seconds\n";
  return;
}

