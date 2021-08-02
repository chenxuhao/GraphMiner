// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>
#include "sgl.h"
std::map<char,double> time_ops;

int main(int argc, char **argv) {
  if(argc < 3) {
    std::cerr << "usage: " << argv[0] << " <graph prefix> <pattern> [num_gpu(1)] [chunk_size(1024)]\n";
    printf("Example: %s /graph_inputs/mico/graph rectangle\n", argv[0]);
    exit(1);
  }
  std::cout << "Subgraph Listing/Counting (undirected graph only)\n";
  Graph g(argv[1]);
  Pattern patt(argv[2]);
  std::cout << "Pattern: " << patt.get_name() << "\n";
  int n_devices = 1;
  int chunk_size = 1024;
  if (argc > 3) n_devices = atoi(argv[3]);
  if (argc > 4) chunk_size = atoi(argv[4]);
  uint64_t h_total = 0;
  auto m = g.num_vertices();
  auto nnz = g.num_edges();
  std::cout << "|V| " << m << " |E| " << nnz << "\n";
  time_ops[OP_INTERSECT] = 0;
  time_ops[OP_DIFFERENCE] = 0;
  SglSolver(g, patt, h_total, n_devices, chunk_size);
  std::cout << "total_num = " << h_total << "\n";
  std::cout << "--------------------\n";
  std::cout << "set intersection time: " << time_ops[OP_INTERSECT] << "\n";
  std::cout << "set difference time: " << time_ops[OP_DIFFERENCE] << "\n";
  return 0;
}
