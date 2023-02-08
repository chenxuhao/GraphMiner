// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "ASAP.h"


void TCSolver(Graph &g, uint64_t &total, int, int, vector<float> args) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  g.init_simple_edgelist();
  std::cout << "OpenMP Triangle Counting (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  uint64_t counter = 0;
  #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
  for (int i = 0; i < int(args[0]); i++) {
      output e1 = sample_edge(g);

      output e2 = sample_out_neighbor_edge(g, e1.v0, e1.v1);

      counter += get_closure(g, e1.v0, e2.v1, 1) * e2.factor * e1.factor;
  }
  printf("count: %lu\n", counter);
  // scale down by number of samples
  total = counter / args[0];
  t.Stop();
  std::cout << "runtime [tc_sample_edge_stream] = " << t.Seconds() << " sec\n";
  return;
}

