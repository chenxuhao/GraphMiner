#include "graph.h"
#include "intersect.h"

void SglSolver(Graph &g, Pattern &p, uint64_t &total, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("OpenMP edge-induced subgraph listing (%d threads) ...\n", num_threads);
  uint64_t counter = 0;
  std::vector<std::vector<uint8_t>> cmaps(num_threads);
  for (int tid = 0; tid < num_threads; tid ++) {
    auto &cmap = cmaps[tid];
    cmap.resize(g.size()); // the connectivity code
    std::fill(cmap.begin(), cmap.end(), 0);
  }
#ifdef PROFILE_LATENCY
  std::vector<uint64_t> nticks(num_threads, 0);
  std::vector<uint64_t> nqueries(num_threads, 0);
#endif

  Timer t;
  t.Start();
  std::cout << "Running the c-map implementation\n";
  if (p.is_house()) {
    #include "house_cmap.h"
  } else if (p.is_pentagon()) {
    #include "pentagon_cmap.h"
  } else if (p.is_rectangle()) {
    #include "rectangle_cmap.h"
  } else {
    #include "diamond_cmap.h"
  }
  total = counter;
  t.Stop();
#ifdef PROFILE_LATENCY
  uint64_t total_query_latency = 0;
  uint64_t total_num_queries = 0;
  for (int tid = 0; tid < num_threads; tid ++) {
    total_query_latency += nticks[tid];
    total_num_queries += nqueries[tid];
  }
  auto avg_query_latency = total_query_latency / total_num_queries;
  std::cout << "average c-map query latency: " <<  avg_query_latency << " cycles\n";
#endif
  std::cout << "runtime = " <<  t.Seconds() << " seconds\n";
  return;
}

