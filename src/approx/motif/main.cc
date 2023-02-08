// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include "graph.h"
#include "pattern.hh"

void MotifSolver(Graph &g, int k, std::vector<uint64_t> &accum, int, int);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << "<graph> [ngpu(0)] [chunk_size(1024)]\n";
    std::cout << "Example: " << argv[0] << " /graph_inputs/mico/graph 4\n";
    exit(1);
  }
  Graph g(argv[1]);
  int n_devices = 1;
  int chunk_size = 1024;
  if (argc > 2) n_devices = atoi(argv[2]);
  if (argc > 3) chunk_size = atoi(argv[3]);
  std::cout << "4-motif counting (only for undirected graphs)\n";
  g.print_meta_data();
 
  int num_patterns = num_possible_patterns[4];
  std::cout << "num_patterns: " << num_patterns << "\n";
  std::vector<uint64_t> total(num_patterns, 0);
  MotifSolver(g, 4, total, n_devices, chunk_size);
  for (int i = 0; i < num_patterns; i++)
    std::cout << "pattern " << i << ": " << total[i] << "\n";
  return 0;
}

