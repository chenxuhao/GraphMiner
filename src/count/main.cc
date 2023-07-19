// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "pattern.hh"

std::map<char,double> time_ops;
void ScSolver(Graph &g, Pattern &p, uint64_t &total, int n_devices, int chunk_size);

int main(int argc, char **argv) {
  if(argc < 3) {
    std::cerr << "usage: " << argv[0] << " <graph prefix> <pattern> [num_gpu(1)] [chunk_size(1024)]\n";
    printf("Example: %s /graph_inputs/mico/graph rectangle\n", argv[0]);
    exit(1);
  }
  std::cout << "Subgraph Counting (undirected graph only)\n";
  Graph g(argv[1]);
  Pattern patt(argv[2]);
  std::cout << "Pattern: " << patt.get_name() << "\n";
  int n_devices = 1;
  int chunk_size = 1024;
  if (argc > 3) n_devices = atoi(argv[3]);
  if (argc > 4) chunk_size = atoi(argv[4]);
  g.print_meta_data();

  time_ops[OP_INTERSECT] = 0;
  time_ops[OP_DIFFERENCE] = 0;
  uint64_t h_total = 0;
  ScSolver(g, patt, h_total, n_devices, chunk_size);
  std::cout << "total_num = " << h_total << "\n";
  //std::cout << "--------------------\n";
  //std::cout << "set intersection time: " << time_ops[OP_INTERSECT] << "\n";
  //std::cout << "set difference time: " << time_ops[OP_DIFFERENCE] << "\n";
  return 0;
}

