#include "graph.h"

void TCSolver(Graph &g, uint64_t &total, int n_gpu, int chunk_size, int threshold);

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout << "Usage: " << argv[0] << " <graph> <use_dag> <degree_threshold> [num_gpu(1)] [chunk_size(1024)] [adj_sorted(1)]\n";
    std::cout << "Example: " << argv[0] << " ../../../inputs/mico/graph 1 350\n";
    exit(1);
  }
  std::cout << "Triangle Counting: we assume the neighbor lists are sorted.\n";
	Graph g(argv[1], atoi(argv[2])); // set DAG
	int n_devices = 1;
  int chunk_size = 1024;
  int adj_sorted = 1;
  if (argc > 4) n_devices = atoi(argv[4]);
  if (argc > 5) chunk_size = atoi(argv[5]);
  g.print_meta_data();
  if (argc > 6) adj_sorted = atoi(argv[6]);
  if (!adj_sorted) g.sort_neighbors();
  uint64_t total = 0;
  TCSolver(g, total, n_devices, chunk_size, atoi(argv[3]));
  std::cout << "total_num_triangles = " << total << "\n";
	return 0;
}

