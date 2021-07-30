// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "fsm.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <filename> [max_size(3)] [min_support(5000)]\n", argv[0]);
    exit(1);
  } 
  Graph g(argv[1], 0, 1);
  unsigned k = 2;
  if (argc > 2) k = atoi(argv[2]);
  unsigned minsup = 5000;
  if (argc > 3) minsup = atoi(argv[3]);
  std::cout << "max_size = " << k << "\n";
  std::cout << "min_support = " << minsup << "\n";

  int m = g.num_vertices();
  int nnz = g.num_edges();
  int nlabels = g.get_vertex_classes();
  std::cout << "Number of unique labels: " << nlabels << "\n";
  std::cout << "|V| " << m << " |E| " << nnz << "\n";
  int num_freqent_patterns = 0;
  FsmSolver(g, k, minsup, nlabels, num_freqent_patterns);
  std::cout << "\nNumber of frequent patterns: " << num_freqent_patterns << "\n";
  return 0;
}

