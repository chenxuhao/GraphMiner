// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include <mpi.h>
#define EVEN_SPLIT

void triangle_warp_edge(int ngpus, int nranks, int rank, Graph &g, uint64_t &total);

void TCSolver(Graph &g, uint64_t &total, int n_gpus, int chunk_size) {
  MPI_Init(NULL, NULL);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  std::cout << processor_name << " is the " << world_rank 
            << " machine in " << world_size << " machines\n";
  std::string head = "[Host " + std::to_string(world_rank) + "] ";

  eidType nnz = g.init_edgelist();
  //std::cout << head << "Total edgelist size = " << nnz << "\n";

  uint64_t local_sum = 0;
  triangle_warp_edge(n_gpus, world_size, world_rank, g, local_sum);
  //std::cout << head << "Local sum = " << local_sum << "\n";

  uint64_t global_sum;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  if (world_rank == 0) std::cout << "Total sum = " << global_sum << "\n";
  total = global_sum;
  MPI_Finalize();
  return;
}

