// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include <set>

#define NUM_SAMPLES 200000

s_edge is_adj(set<s_edge>* edges, s_edge e) {
  for (auto itr = (*edges).begin();
       itr != (*edges).end(); itr++)
  {
     if(itr->dst == e.src) {
        edges->erase(*itr);
        return s_edge{e.dst, itr->src};
     }
     if(itr->src == e.dst) {
        edges->erase(*itr);
        return s_edge{itr->dst, e.src};
     }
  }
  return s_edge{0,0};
}

void TCSolver(Graph &g, uint64_t &total, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  g.create_edge_stream();
  std::cout << "OpenMP Triangle Counting (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  uint64_t counter = 0;
  set<s_edge> sampled_edges;
  set<s_edge> awaiting_closing_edges;
  
  #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
  for (int i = 0; i < g.E(); i++) {
    s_edge e = g.stream_edge(i);
    if(awaiting_closing_edges.count(e)) {
      counter += awaiting_closing_edges.count(e);
      awaiting_closing_edges.erase(e);
    }

    if(rand() % g.E() < g.E() / NUM_SAMPLES) {
      sampled_edges.insert(e);
    }

    auto adj = is_adj(&sampled_edges,e); //will only be true once
    // scale by c?
    if(!(adj.src == 0 && adj.dst == 0)) {
      awaiting_closing_edges.insert(adj);
    }

  }
  printf("count: %lu\n", counter);
  // scale down by number of samples
  total = counter / NUM_SAMPLES;
  t.Stop();
  std::cout << "runtime [tc_sample_edge_stream] = " << t.Seconds() << " sec\n";
  return;
}

