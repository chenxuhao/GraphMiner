// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include "graph.h"
#include "cmap.h"
#include "clique_base.h"

void cmap_kclique(Graph &g, int k, uint64_t &total);

void CliqueSolver(Graph &g, int k, uint64_t &total, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP " << k << "-clique listing (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  double start_time = omp_get_wtime();
  if (k==3) automine_3clique(g, total);
  else cmap_kclique(g, k, total);
  double run_time = omp_get_wtime() - start_time;
  t.Stop();
  std::cout << "runtime [omp_recursive] = " << run_time << " sec\n";
  return;
}

void extend_clique(int level, int k, Graph &g, 
                   std::vector<VertexSet> &vertices, 
                   cmap8_t &cmap, uint64_t &counter) {
  if (level == k - 2) {
    uint64_t local_counter = 0;
    for (int i = 0; i < vertices[level-2].size(); i++) {
      auto v = vertices[level-2][i];
      for (auto u : g.N(v)) {
        //if (u >= v) break;
        if (cmap.get(u) == level)
          local_counter ++;
      }
    }
    counter += local_counter;
    return;
  }
  for (int i = 0; i < vertices[level-2].size(); i++) {
    vertices[level-1].clear();
    auto v = vertices[level-2][i];
    for (auto u : g.N(v)) {
      //if (u >= v) break;
      if (cmap.get(u) == level) {
        vertices[level-1].add(u);
        cmap.set(u, level+1);
      }
    }
    extend_clique(level+1, k, g, vertices, cmap, counter);
    for (auto w : vertices[level-1]) cmap.set(w, level);
  }
}

void cmap_kclique(Graph &g, int k, uint64_t &total) {
  assert(k>3);
  uint64_t counter = 0;
  #pragma omp parallel
  {
  std::vector<VertexSet> vertices(k-2);
  cmap8_t cmap;
  cmap.init(g.size());
  #pragma omp for schedule(dynamic, 1) reduction(+:counter)
  for (vidType v0 = 0; v0 < g.V(); v0 ++) {
    for (auto w : g.N(v0)) cmap.set(w, 1);
    for (auto v1 : g.N(v0)) {
      vertices[0].clear();
      for (auto w : g.N(v1)) {
        if (cmap.get(w) == 1) {
          cmap.set(w, 2);
          vertices[0].add(w);
        }
      }
      extend_clique(2, k, g, vertices, cmap, counter);
      for (auto w : vertices[0]) cmap.set(w, 1);
    }
    for (auto w : g.N(v0)) cmap.set(w, 0);
  }
  }
  total = counter;
}

