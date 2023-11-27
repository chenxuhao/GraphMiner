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
  if (p.is_clique()) {
    //#include "clique.h"
    std::cout << "Please go to clique counting\n";
  // 4-motifs
  } else if (p.is_rectangle()) {
    #include "rectangle.h"
  } else if (p.is_tailedtriangle()) {
    #include "tailedtriangle.h"
  } else if (p.is_diamond()) {
    #include "diamond.h"
  } else if (p.is_4path()) { // a.k.a, 4-chain
    #include "4path.h"
  } else if (p.is_3star()) {
    #include "3star.h"
  // 5-motifs
  } else if (p.is_5path()) { // a.k.a, 5-chain
    #include "5path.h"
  } else if (p.is_pentagon()) { // a.k.a, 5-cycle
    #include "pentagon.h"
  } else if (p.is_house()) {
    #include "house.h"
  } else if (p.is_semihouse()) {
    #include "semihouse.h"
  } else if (p.is_closedhouse()) {
    #include "closedhouse.h"
  } else if (p.is_hourglass()) {
    #include "hourglass.h"
  } else if (p.is_taileddiamond()) {
    #include "taileddiamond.h"
  } else if (p.is_taileddiamond2()) {
    #include "taileddiamond2.h"
  // 6-motifs
  } else if (p.is_6path()) { // a.k.a, 6-chain
    #include "6path.h"
  } else if (p.is_dumbbell()) {
    #include "dumbbell.h"
  } else {
    std::cout << "Not implemented\n";
  }
  total = counter;
  t.Stop();
  std::cout << "runtime = " <<  t.Seconds() << " seconds\n";
  return;
}

