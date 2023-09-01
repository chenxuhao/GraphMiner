#include "graph.h"
#include "pattern.hh"

std::map<char,double> time_ops;
void HouseSolver(Graph &g, uint64_t& total, int, int, int threshold);
void PentagonSolver(Graph &g, uint64_t& total, int, int, int threshold);
void RectangleSolver(Graph &g, uint64_t& total, int, int, int threshold);
void DiamondSolver(Graph &g, uint64_t& total, int, int, int threshold);

int main(int argc, char **argv) {
  if(argc < 4) {
    std::cout << "Usage: " << argv[0] << " <graph prefix> <pattern> <threshold> [num_gpu(1)] [chunk_size(1024)]\n";
    std::cout << "Example: " << argv[0] << "../../../inputs/mico/graph diamond 400\n";
    exit(1);
  }
  std::cout << "Subgraph Listing/Counting (undirected graph only)\n";
  Graph g(argv[1]);
  Pattern patt(argv[2]);
  std::cout << "Pattern: " << patt.get_name() << "\n";
  int threshold = atoi(argv[3]);
  int n_devices = 1;
  int chunk_size = 1024;
  if (argc > 4) n_devices = atoi(argv[4]);
  if (argc > 5) chunk_size = atoi(argv[5]);
  g.print_meta_data();

  time_ops[OP_INTERSECT] = 0;
  time_ops[OP_DIFFERENCE] = 0;
  uint64_t total = 0;
  if (patt.is_house()) {
    HouseSolver(g, total, threshold, n_devices, chunk_size);
  } else if (patt.is_pentagon()) {
    PentagonSolver(g, total, threshold, n_devices, chunk_size);
  } else if (patt.is_rectangle()) {
    RectangleSolver(g, total, threshold, n_devices, chunk_size);
  } else if (patt.is_diamond()){
    DiamondSolver(g, total, threshold, n_devices, chunk_size);
  } else {
    std::cout << "Error: invalid pattern\n";
    exit(1);
  }
  // SglSolver(g, patt, h_total, n_devices, chunk_size);
  std::cout << "total_num = " << total << "\n";
  return 0;
}