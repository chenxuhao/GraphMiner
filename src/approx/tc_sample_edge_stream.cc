// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include <set>
#include <unordered_map>

#define NUM_SAMPLES 200000

s_edge is_adj(unordered_map<s_edge, int> edges, s_edge e)
{
  for( const std::pair<const s_edge, int>& n : edges ) {
     s_edge edge = n.first;
     int count = n.second;

     // incoming edge adj. fount
     if(edge.src == e.dst) {
      edges[edge] += 1;
      if(count == 0) {
        return s_edge{edge.dst, e.src};
      }
     }

     //outgoing edg. adj. found
     if(edge.dst == e.src) {
      edges[edge] += 1;
      if(count == 0) {
        s_edge{e.dst, edge.src};
      }
     }
  }
  return s_edge{0,0};
}

void TCSolver(Graph &g, uint64_t &total, int, int)
{
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
  std::unordered_map<s_edge, int> sampled_edges;
  set<s_edge> awaiting_closing_edges;

  //#pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
  for (int i = 0; i < g.E(); i++)
  {
    s_edge e = g.stream_edge(i);
    if (awaiting_closing_edges.count(e))
    {
      counter += awaiting_closing_edges.count(e);
      awaiting_closing_edges.erase(e);
    }

    if (rand() % g.E() < g.E() / NUM_SAMPLES)
    {
      // printf("added edge: %d,%d\n",e.src, e.dst);
      sampled_edges[e] = 0;
    }

    auto adj = is_adj(sampled_edges, e); // will only be true once
    if (!(adj.src == 0 && adj.dst == 0))
    {
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
