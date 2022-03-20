#include "graph.h"

void SglSolver(Graph &g, Pattern &p, uint64_t &total, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("OpenMP edge-induced subgraph listing (%d threads) ...\n", num_threads);
  uint64_t counter = 0;
  Timer t;
  t.Start();
  std::cout << "Running the AutoMine/GraphZero implementation\n";
  if (p.is_house()) {
    #include "house_zero.h"
  } else if (p.is_pentagon()) {
    #include "pentagon_zero.h"
  } else if (p.is_rectangle()) {
    #include "rectangle_zero.h"
  } else {
    #include "diamond_zero.h"
  }
  total = counter;
  t.Stop();
  std::cout << "runtime = " <<  t.Seconds() << " seconds\n";
  return;
}

