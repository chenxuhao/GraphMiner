// Copyright 2021 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "gks.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <filename> [max_num_vertices(3)]\n";
    std::cout << "Example: ./gks_omp_base ../input/citeseer/graph 5\n";
    exit(1);
  } 
  int k = 3;
  if (argc > 2) k = atoi(argv[2]);
  std::cout << "Graph Keyword Search, ";
  std::cout << "max_size: " << k << ", ";
  int n_devices = 1;
  int chunk_size = 1024;
  if (argc > 3) n_devices = atoi(argv[3]);
  if (argc > 4) chunk_size = atoi(argv[4]);
  Keywords kws(1, 2, 3);

  Graph g(argv[1], 0, 1);
  auto m = g.num_vertices();
  auto nnz = g.num_edges();
  int nlabels = g.get_vertex_classes();
  std::cout << "|V| " << m << " |E| " << nnz << "\n";
  std::cout << "Number of unique labels: " << nlabels << "\n";
  uint64_t num = 0;
  GksSolver(g, k, nlabels, kws, num, n_devices, chunk_size);
  std::cout << "Number of matches: " << num << "\n";
  return 0;
}

