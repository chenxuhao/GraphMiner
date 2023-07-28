// Copyright 2020, MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void TCSolver(Graph &g, uint64_t &total, int n_gpu, int chunk_size);

int main(int argc, char *argv[]) {
  srand ( time(NULL) );
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> [num_gpu(1)] [chunk_size(1024)] [adj_sorted(1)]\n";
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
  uint64_t total = 0;
  g.sample_tree(9);

  TCSolver(g, total, n_devices, chunk_size);
  std::cout << "total_num_pattern = " << total << "\n";
  return 0;
}

