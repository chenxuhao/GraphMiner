// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include "graph.h"
#include "string.h"

void CliqueSolver(Graph &g, int k, uint64_t &total, int threshold, int, int);

int main(int argc, char *argv[]) {
	if (argc < 4) {
		std::cout << "Usage: " << argv[0] << "<graph> <k> <threshold> [ngpu(0)] [chunk_size(1024)]\n";
		std::cout << "Example: " << argv[0] << " ../../../inputs/mico/graph 4 400\n";
		exit(1);
	}
	std::cout << "k-clique listing with undirected graphs\n";
	// if (USE_DAG) std::cout << "Using DAG (static orientation)\n";
	// Graph g(argv[1], USE_DAG); // use DAG
	Graph g(argv[1], false); // do not use DAG by default
	
	int k = atoi(argv[2]);
	int threshold = atoi(argv[3]);
	int n_devices = 1;
	int chunk_size = 1024;
	if (argc > 4) n_devices = atoi(argv[4]);
	if (argc > 5) chunk_size = atoi(argv[5]);
	g.print_meta_data();
	
	uint64_t total = 0;
	CliqueSolver(g, k, total, threshold, n_devices, chunk_size);
  	std::cout << "num_" << k << "-cliques = " << total << "\n";
	return 0;
}

