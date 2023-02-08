// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu> and Anna Arpaci-Dusseau <annaad@mit.edu>
#include "graph.h"
#define NUMSAMPLES 100000

#include <algorithm>


struct mask {
  int start;
  int end;
  int num;
  vector<int> indexes;
};

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

struct mask get_mask(int start, int end, int num) {
  struct mask m;
  m.start = start;
  m.end = end;
  m.num = num;

  while(m.indexes.size() < num) {
    m.indexes.push_back(rand() % (end - start) + start);
  }

  return m;
}

struct mask get_mask_weighted(int start, int end, int num, Graph &g) {
  struct mask m;
  m.start = start;
  m.end = end;
  m.num = num;

  vector<int> weighted_options;
  int total = 0;

  for(auto i = start; i < end; i++) {
    for(int j = 0; j < g.N(i).size(); j++) {
      weighted_options.push_back(i);
    }
  }

  std::random_shuffle(weighted_options.begin(), weighted_options.end());

  for(int i = 0; i < num; i++){
    m.indexes.push_back(weighted_options[i]);
  }

  return m;
}


void TCSolver(Graph &g, uint64_t &total, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  Timer t;
  struct mask m = get_mask_weighted(0, g.V(), NUMSAMPLES, g);
  t.Start();
  uint64_t counter = 0;
  #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
  for (vidType u = 0; u < m.num; u ++) {
    int l0 = m.indexes[u];

    if(g.N(l0).size() == 0) {
      continue;
    }
    // finish out edge
    auto l1 = g.N(l0)[rand() % g.N(l0).size()];

    if(g.N(l1).size() > 0) {
      // get a neighbor of l1, vertex l2
      // 1/c prob of choosing some neighbor
      auto l2 = g.N(l1)[rand() % g.N(l1).size()];
      
      // TODO: bug here?
      assert(l2 != l0);
      // shouldn't happen with DAG?
      assert(!(contains(g.N(l0), l2) && contains(g.N(l2), l0)));  

      // check if either neighbor edge completes triangle, if so incr by m*c
      counter += (contains(g.N(l0), l2) || contains(g.N(l2), l0)) * g.N(l1).size() * g.E();
    }
  }
  total = counter / NUMSAMPLES;
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}


