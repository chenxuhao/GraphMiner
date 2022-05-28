// Copyright 2020, MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void TCSolver(Graph &g, uint64_t &total, int n_gpu, int chunk_size);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> [num_gpu(1)] [chunk_size(1024)]\n";
    std::cout << "Example: " << argv[0] << " /graph_inputs/mico/graph\n";
    exit(1);
  }
  std::cout << "Triangle Counting\n";
  //if (USE_DAG) std::cout << "Using DAG (static orientation)\n";
  Graph g(argv[1], USE_DAG); // use DAG
  int n_devices = 1;
  int chunk_size = 1024;
  if (argc > 2) n_devices = atoi(argv[2]);
  if (argc > 3) chunk_size = atoi(argv[3]);
  g.print_meta_data();
  uint64_t total = 0;
  TCSolver(g, total, n_devices, chunk_size);
  std::cout << "total_num_triangles = " << total << "\n";
  return 0;
}

