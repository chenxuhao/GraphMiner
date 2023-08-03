// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu> and Anna Arpaci-Dusseau <annaad@mit.edu>
#include "graph.h"
#define NUMSAMPLES 100000

struct mask {
  int start;
  int end;
  int num;
  vector<int> indexes;
};

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


void TCSolver(Graph &g, uint64_t &total, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  Timer t;
  struct mask m = get_mask(0, g.V(), NUMSAMPLES);
  t.Start();
  uint64_t counter = 0;
  #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
  for (vidType u = 0; u < m.num; u ++) {
    int v0 = m.indexes[u];
    
    // sample neighbor edge 1/sf chance
    auto adj_v = g.N(v0); 
    int sf = adj_v.size();
    if(sf == 0) continue;
    struct mask m2 = get_mask(0, sf, 1);
    vidType v1 = adj_v[m2.indexes[0]];
   
    // look for closing edge exactly
    auto adj_v1 = g.N(v1);  
    counter += (uint64_t)intersection_num(adj_v1, g.N(v0))*sf*g.V();  //scale up by m*c
  }
  total = counter / NUMSAMPLES;
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}


