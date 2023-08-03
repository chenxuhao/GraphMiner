// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

#define NUM_SAMPLES 500000

bool contains(VertexSet n, vidType v) {
  for(vidType i = 0; i < n.size(); i++) {
    if(n[i] == v) {
      return true;
    }
    if(n[i] > v) {
      return false;
    }
  }
  return false;
}

void TCSolver(Graph &g, uint64_t &total, int, int, vector<float> args) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  g.init_simple_edgelist();
  std::cout << "OpenMP 4-Cycle Counting (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  uint64_t counter = 0;
  #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
  for (int i = 0; i < int(args[0]); i++) {
    // pick one edge uniformly at random, [l0, l1]
    // 1 / m prob of choosing an edge
    eidType randE = rand() % g.E();

    auto l0 = g.get_src(randE);
    auto l1 = g.get_dst(randE);

    assert(contains(g.N(l0), l1));
    
    // TODO: this fails...
    // dst degree > src degree, or they are equal and dst ID > src Id
    //assert(g.get_degree(l1) > g.get_degree(l0) || (g.get_degree(l1) == g.get_degree(l0) && l1 > l0)); 

    // consider from given edge, case , only coming after (outgoing) 
    // l0 -> l1, l1 -> l2,  l0 -> l2

    if(g.N(l1).size() > 0) {
      // get a neighbor of l1, vertex l2
      // 1/c prob of choosing some neighbor
      auto l2 = g.N(l1)[rand() % g.N(l1).size()];
      
      // TODO: bug here?
      assert(l2 != l0);
      // shouldn't happen with DAG?
      assert(!(contains(g.N(l0), l2) && contains(g.N(l2), l0)));  

      
      counter += intersection_num(g.N(l2), g.N(l0)) * g.N(l1).size() * g.E();
    }

  }
  printf("count: %lu\n", counter);
  // scale down by number of samples
  total = counter / args[0];
  t.Stop();
  std::cout << "runtime [tc_sample_edge_stream] = " << t.Seconds() << " sec\n";
  return;
}

