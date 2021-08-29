// Copyright 2021 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "query.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <filename> \n";
    std::cout << "Example: ./query_omp_base ../input/citeseer/graph ../input/q1-triangle.graph\n";
    exit(1);
  }
  Pattern p(argv[2], 1);
  int n_devices = 1;
  int chunk_size = 1024;
  if (argc > 3) n_devices = atoi(argv[3]);
  if (argc > 4) chunk_size = atoi(argv[4]);
  auto ncolors = p.get_num_labels();
  std::cout << "Labeled Graph Querying, ";
  std::cout << "Pattern: " << ncolors << "-color " << p.get_name() << " " << p << "\n";

  Graph g(argv[1], 0, 1);
  auto m = g.num_vertices();
  auto nnz = g.num_edges();
  int nlabels = g.get_vertex_classes();
  std::cout << "Data Graph |V| " << m << " |E| " << nnz << "\n";
  std::cout << "Maximum degree: " << g.get_max_degree() << "\n";
  std::cout << "Number of unique labels: " << nlabels << "\n";
  uint64_t num = 0;
  QuerySolver(g, p, num, n_devices, chunk_size);
  std::cout << "Number of matches: " << num << "\n";
  return 0;
}

