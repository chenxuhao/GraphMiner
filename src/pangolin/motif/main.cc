// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "motif.h"
static int num_patterns[3] = {2, 6, 21};

void printout_motifs(std::vector<uint64_t> &counters) {
  if (counters.size() == 2) {
    std::cout << "triangles\t" << counters[0] << std::endl;
    std::cout << "3-chains\t" << counters[1] << std::endl;
  } else if (counters.size() == 6) {
    std::cout << "4-paths --> " << counters[0] << std::endl;
    std::cout << "3-stars --> " << counters[1] << std::endl;
    std::cout << "4-cycles --> " << counters[2] << std::endl;
    std::cout << "ailed-triangles --> " << counters[3] << std::endl;
    std::cout << "diamonds --> " << counters[4] << std::endl;
    std::cout << "4-cliques --> " << counters[5] << std::endl;
  } else {
    std::cout << "too many patterns to show\n";
  }
  std::cout << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << "<graph> [max_size(3)]\n";
    std::cout << "Example: ./bin/pangolin/" << argv[0] << " ./inputs/citeseer/graph 3\n";
    exit(1);
  } 
  Graph g(argv[1], 0);
  unsigned k = 3;
  if (argc == 3) k = atoi(argv[2]);
  int npatterns = num_patterns[k-3];
  std::cout << k << "-motif has " << npatterns << " patterns in total\n";
  std::vector<uint64_t> counters(npatterns);
  for (int i = 0; i < npatterns; i++) counters[i] = 0;
  int m = g.num_vertices();
  int nnz = g.num_edges();
  std::cout << "|V| " << m << " |E| " << nnz << "\n";
  MotifSolver(g, k, counters);
  printout_motifs(counters);
  return 0;
}

