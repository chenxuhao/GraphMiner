// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include <mpi.h>

void TCSolver(Graph &g, uint64_t &total, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }

  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  if (world_rank == 0) {
    std::cout << "MPI OpenMP TC: " << world_size << " machines, " << num_threads << " threads per machine\n";
  }
  auto num_tasks = g.V();
  vidType ntasks_per_rank = num_tasks / world_size;
  vidType begin = ntasks_per_rank * world_rank;
  vidType end = ntasks_per_rank * (world_rank+1);
  if (end > num_tasks) end = num_tasks;
  std::cout << "Machine " << world_rank << " " << processor_name 
            << ": [" << begin << ", " << end << ")\n";

  Timer t;
  t.Start();
  uint64_t counter = 0;
  #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
  for (auto u = begin; u < end; u ++) {
    auto yu = g.N(u);
    for (auto v : yu) {
      counter += (uint64_t)intersection_num(yu, g.N(v));
    } 
  }
  t.Stop();
  std::cout << "runtime [dist_cpu] = " << t.Seconds() << " sec\n";
  std::cout << "Local sum = " << counter << " on machine " << world_rank << "\n";

  uint64_t global_sum;
  //MPI_Reduce(&counter, &global_sum, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Allreduce(&counter, &global_sum, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  if (world_rank == 0)
    std::cout << "Total sum = " << global_sum << "\n";
  total = global_sum;

  // Finalize the MPI environment.
  MPI_Finalize();
  return;
}

