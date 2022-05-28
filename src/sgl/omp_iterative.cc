// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
void search(Graph &g, int k, VertexList ops, VertexLists ancestors, uint64_t &total);
void search_cmap(Graph &g, int k, uint64_t &total);

void SglSolver(Graph &g, Pattern &p, uint64_t &total, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP " << "Subgraph Listing (" << num_threads << " threads)\n";
  auto k = p.size();
  double start_time = omp_get_wtime();
  VertexList ops(k-2);
  ops[0] = 0; ops[1] = 1;
  VertexLists ancestors(k-2);
  for (int i = 0; i < k-2; i++)
    ancestors[i].resize(2);
  ancestors[0][0] = 0;
  ancestors[0][1] = -1;
  ancestors[1][0] = 1;
  ancestors[1][1] = 2;
  search(g, k, ops, ancestors, total);
  double run_time = omp_get_wtime() - start_time;
  std::cout << "runtime [omp_base] = " << run_time << " sec\n";
  return;
}

void search(Graph &g, int k, VertexList ops, VertexLists ancestors, uint64_t &total) {
  assert(k > 2);
  uint64_t counter = 0;
  #pragma omp parallel
  {
  std::vector<VertexSet> vertices(k-2);
  std::vector<vidType> idx(k-2);
  std::vector<vidType*> ptrs(k-2);
  std::vector<vidType> sizes(k-2);
  std::vector<vidType> stack(k);
  Status state = Idle;
  #pragma omp for schedule(dynamic, 1) reduction(+:counter)
  for (vidType v0 = 0; v0 < g.size(); v0 ++) {
    int depth = 0;
    stack[0] = v0;
    state = Extending;
    for (auto v1 : g.N(v0)) {
      stack[1] = v1;
      depth = 1;
      state = Extending;
      while (1) {
        auto p0 = ancestors[depth][0];
        auto u = stack[p0];
        if (depth == k-2) { // found a match
          if (ops[depth-1] == 1) { // intersection
            auto p1 = ancestors[depth][1];
            auto v = stack[p1];
            counter += intersection_num(g.N(u), g.N(v));
          } else {
            counter += g.get_degree(u); // TODO: avoid repetitive vertices
          }
          depth --; // backtrack
          state = IteratingEdge;
        } else if (state == Extending) {
          vertices[depth-1].clear();
          if (ops[depth] == 1) { // intersection
            auto p1 = ancestors[depth][1];
            auto v = stack[p1];
            intersection(g.N(u), g.N(v), vertices[depth-1]);
            sizes[depth-1] = vertices[depth-1].size();
            ptrs[depth-1] = vertices[depth-1].data();
          } else ptrs[depth-1] = g.N(u).data();
          idx[depth-1] = 0;
        }
        if (depth == 0) break; 
        if (idx[depth-1] == sizes[depth-1]) {
          if (depth == 1) break; // this subtree is done
          else { // backtrack
            depth --;
            state = IteratingEdge;
          }
        } else {
          auto i = idx[depth-1];
          auto w = ptrs[depth-1][i];
          idx[depth-1] = i + 1;
          depth ++;
          stack[depth] = w;
          state = Extending;
        }
      }
    } // end while
  } // end for
  } // end omp
  total = counter;
}

