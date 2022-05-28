// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include "grpah.h"
#include "cmap.h"
void kclique(Graph &g, int k, uint64_t &total);
void kclique_cmap(Graph &g, int k, uint64_t &total);

void CliqueSolver(Graph &g, int k, uint64_t &total, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP " << k << "-clique listing (" << num_threads << " threads)\n";
  double start_time = omp_get_wtime();
  //kclique(g, k, total);
  kclique_cmap(g, k, total);
  double run_time = omp_get_wtime() - start_time;
  std::cout << "runtime [omp_iterative] = " << run_time << " sec\n";
  return;
}

void kclique(Graph &g, int k, uint64_t &total) {
  assert(k > 2);
  uint64_t counter = 0;
  #pragma omp parallel
  {
  std::vector<VertexSet> vertices(k-2);
  std::vector<vidType> idx(k-2);
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
      while(1) {
        if (depth == k-2) { // found a match
          if (depth >= 2) {
            auto u = stack[depth];
            counter += intersection_num(vertices[depth-2], g.N(u));
          } else {
            assert(depth == 1);
            counter += intersection_num(g.N(v0), g.N(v1));
          }
          depth --; // backtrack
          state = IteratingEdge;
        } else if (state == Extending) {
          vertices[depth-1].clear();
          if (depth >= 2) {
            auto u = stack[depth];
            intersection(vertices[depth-2], g.N(u), vertices[depth-1]);
          } else {
            intersection(g.N(v0), g.N(v1), vertices[depth-1]);
          }
          idx[depth-1] = 0;
        }
        if (depth == 0) break; 
        if (idx[depth-1] == vertices[depth-1].size()) {
          if (depth == 1) break; // this subtree is done
          else { // backtrack
            depth --;
            state = IteratingEdge;
          }
        } else {
          auto i = idx[depth-1];
          auto w = vertices[depth-1][i];
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

void kclique_cmap(Graph &g, int k, uint64_t &total) {
  assert(k > 2);
  uint64_t counter = 0;
  #pragma omp parallel
  {
  cmap8_t cmap;
//#ifdef IDENT_CMAP
    cmap.init(g.size());
//#else
//    cmap.init(max_degree);
//#endif
  std::vector<VertexSet> vertices(k-2);
  std::vector<vidType> idx(k-2);
  std::vector<vidType> stack(k);
  //std::vector<vidType> extender(k);
  Status state = Idle;
  #pragma omp for schedule(dynamic, 1) reduction(+:counter)
  for (vidType v0 = 0; v0 < g.size(); v0 ++) {
    //auto tid = omp_get_thread_num();
    int depth = 0;
    stack[0] = v0;
    state = Extending;
    // mark v0's neighbors
    for (auto w : g.N(v0)) {
      //#if USE_DAG == 0
      //if (w >= v0) break;
      //#endif
      cmap.set(w, 1);
    }
    for (auto v1 : g.N(v0)) {
      stack[1] = v1;
      depth = 1;
      state = Extending;
      while(1) {
        if (depth == k-2) { // found a match
          for (auto w : g.N(stack[depth])) {
            if (cmap.get(w) == depth)
              counter ++;
          }
          depth --; // backtrack
          //stack.pop_back();
          state = IteratingEdge;
        } else if (state == Extending) {
          auto u = stack[depth];
          //extender[depth] = u;
          idx[depth-1] = 0;
          vertices[depth-1].clear();
          for (auto w : g.N(u)) {
            if (cmap.get(w) == depth) {
              if (depth < k-2) cmap.set(w, depth+1);
              vertices[depth-1].add(w);
            }
          }
        }
        if (depth == 0) break; 
        if (idx[depth-1] == vertices[depth-1].size()) {
          for (auto w : vertices[depth-1]) cmap.set(w, depth);
          if (depth == 1) break; // this subtree is done
          else { // backtrack
            depth --;
            //stack.pop_back();
            state = IteratingEdge;
          }
        } else {
          auto i = idx[depth-1];
          //auto w = g.getEdgeDst(begin+i); // w is the i-th neighbor of u
          auto w = vertices[depth-1][i];
          idx[depth-1] = i + 1;
          //#if USE_DAG
          //if (cmap.get(w) == depth) {
          //#else
          //if (cmap.get(w) == depth && w < u) {
          //#endif
            //cmap.set(w, depth+1);
            depth ++;
            stack[depth] = w;
            state = Extending;
          //} else {
          //  state = IteratingEdge; // go to next edge
          //}
        }
      }
    } // end while
    //stack.pop_back(); // go to the next v1
    for (auto w : g.N(v0)) cmap.set(w, 0);
  } // end for
  } // end omp
  total = counter;
}

