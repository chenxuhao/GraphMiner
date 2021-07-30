// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include "kcl.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: %s <graph> <k>\n", argv[0]);
    exit(1);
  }
  std::cout << "k-clique listing (BFS exploration)\n";
  Graph g(argv[1], 1); // use DAG
  unsigned k = atoi(argv[2]);
  auto m = g.size();
  auto nnz = g.sizeEdges();
  std::cout << "|V| " << m << " |E| " << nnz << "\n";
  uint64_t total = 0;
  KclSolver(g, k, total);
  //KCLVerifier(g, k, total);
  std::cout << "\ntotal_num_cliques = " << total << "\n\n";
  return 0;
}

