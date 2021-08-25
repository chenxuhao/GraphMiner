// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include "kcl.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << "<graph> <k>\n";
    std::cout << "Example: ./bin/pangolin/" << argv[0] << " ./inputs/citeseer/graph 4\n";
    exit(1);
  }
  unsigned k = atoi(argv[2]);
  std::cout << k << "-clique Listing with undirected graphs\n";
  Graph g(argv[1], 1); // use DAG
  auto m = g.size();
  auto nnz = g.sizeEdges();
  std::cout << "|V| " << m << " |E| " << nnz << "\n";
  uint64_t total = 0;
  KclSolver(g, k, total);
  std::cout << "\ntotal_num_cliques = " << total << "\n\n";
  return 0;
}

