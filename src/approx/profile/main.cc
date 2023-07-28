// Copyright 2020, MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

#define THRESHOLD 3
void TCSolver(Graph &g, uint64_t &total, int c);
void TCSolver_dense(Graph &g, uint64_t &total, int k, int c);
void TCSolver_sparse(Graph &g, uint64_t &total, int k);

int main(int argc, char *argv[]) {
  srand ( time(NULL) );
  if (argc < 5) {
    std::cout << "Usage: " << argv[0] << " <graph> [num_gpu(1)] [chunk_size(1024)] [adj_sorted(1)] [subgraph_profile(0)] [avg_degree_threshold(0)]\n";
    std::cout << "Example: " << argv[0] << " /graph_inputs/mico/graph\n";
    exit(1);
  }
  std::cout << "Approximate Counting: assuming the neighbor lists are sorted.\n";
  Graph g(argv[1], USE_DAG); // use DAG
  //g.print_graph();
  int n_devices = 1;
  int chunk_size = 1024;
  int adj_sorted = 1;


  if (argc > 2) n_devices = atoi(argv[2]);
  if (argc > 3) chunk_size = atoi(argv[3]);
  g.print_meta_data();
  if (argc > 4) adj_sorted = atoi(argv[4]);
  if (!adj_sorted) g.sort_neighbors();
  std::string mode(argv[5]);
  std::vector<std::string> argList(argv + 6, argv + argc);
  

  std::cout <<  mode << "\n";

  if(mode == std::string("color")) {
      std::cout << "|e| before sampling " << g.E() << "\n";
      int c = std::stoi(argList[0]);
      std::cout <<  "coloring graph with fast mode and c =" << c << "\n";
      g.color_sparsify_fast(c);
      std::cout <<  "|e| after sampling " << g.E() << "\n";

      uint64_t total;
      TCSolver(g, total, c);
      std::cout << "TC count = " << total << "\n";
  }
  
  

  // if(subgraph_profile) {
  //     g.sample_tree_subgraph(threshold);
  //     uint64_t total;
  //     TCSolver_thresh(g, total,threshold);
  //     std::cout << "exact sparse count = " << total << "\n";
  //     g.color_sparsify(30); // only sparsifies across nodes > threshold.
  //     uint64_t dtotal;
  //     TCSolver(g, dtotal, threshold,30);
  //     std::cout << "combined count = " << dtotal + total << "\n";
  // } else {
  //     g.color_sparsify(threshold);
  //     g.sample_tree(threshold);
  // }

  // g.sample_tree(threshold);
  // g.sample_tree_subgraph(threshold_s);

  // int total = g.get_intersect_threshold(threshold, threshold_s);

  // std::cout << "num vertices match = " << total << "\n";
  return 0;
}

void TCSolver(Graph &g, uint64_t &total, int c) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Triangle Counting (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  uint64_t counter = 0;
  #pragma omp parallel for reduction(+ : counter)
  for (vidType u = 0; u < g.V(); u ++) {
    auto adj_u = g.N(u);
    for (auto v : adj_u) {
      counter += (uint64_t)intersection_num(adj_u, g.N(v));
    }
  }
  total = counter * (c * c); // assumes that p(the other two edges exist) = (1/c*1/c)
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}

void TCSolver_dense(Graph &g, uint64_t &total, int k, int c) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Triangle Counting (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  uint64_t counter = 0;
  #pragma omp parallel for reduction(+ : counter)
  for (vidType u = 0; u < g.V(); u ++) {
    if(g.get_threshold_s(u) < k) {continue;}
    auto adj_u = g.N(u);
    for (auto v : adj_u) {
      counter += (uint64_t)intersection_num(adj_u, g.N(v));
    }
  }
  total = counter * (c * c); // assumes that p(the other two edges exist) = (1/c*1/c)
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}


void TCSolver_sparse(Graph &g, uint64_t &total, int k) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Triangle Counting (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  uint64_t counter = 0;
  #pragma omp parallel for reduction(+ : counter) schedule(static,256)
  for (vidType u = 0; u < g.V(); u ++) {
    if(g.get_threshold_s(u) >= k) {continue;}
    auto adj_u = g.N(u);
    for (auto v : adj_u) {
      counter += (uint64_t)intersection_num(adj_u, g.N(v));
    }
  }
  total = counter;
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}

