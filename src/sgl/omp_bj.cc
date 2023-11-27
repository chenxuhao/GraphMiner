#include "graph.h"
#include "pattern.hh"
#include "intersect.h"

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
  std::cout << "Running the binary join (decomposition) implementation\n";
  if (p.is_rectangle()) {
    #include "rectangle_bj.h"
  } else if (p.is_house()) {
    #include "house_bj.h"
  } else if (p.is_diamond()) {
    #include "diamond_bj.h"
  //} else if (p.is_pentagon()) {
  //  #include "pentagon_bj.h"
  } else if (p.is_hourglass()) {
    #include "hourglass_bj.h"
  } else if (p.is_dumbbell()) {
    #include "dumbbell_bj.h"
  } else {
    std::cout << "Not implemented\n";
  }
  total = counter;
  t.Stop();
  std::cout << "runtime = " <<  t.Seconds() << " seconds\n";
  return;
}

