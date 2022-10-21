#include "graph.h"
#include "filter.h"
#include "query_plan.h"

void QuerySolver(Graph &g, Pattern &p, uint64_t &total, bool use_filter, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  uint64_t counter = 0;
  int nlabels = g.get_vertex_classes();
  if (nlabels < 3) return;
  p.buildCoreTable();
  p.BuildNLF();
  g.BuildNLF();
  std::cout << "OpenMP Graph Querying (" << num_threads << " threads)\n";

  Timer t;
  t.Start();
  if (use_filter) {
    Filter filter("CFL");
    VertexList candidates_count(p.size(), 0);
    VertexLists candidates(p.size());
    auto max_num = g.get_max_label_frequency();
    for (vidType i = 0; i < p.size(); ++i)
      candidates[i].resize(max_num);

    // filter invalid candidates
    std::cout << "Start filtering...\n";
    double t1 = omp_get_wtime();
    filter.filtering(&g, &p, candidates, candidates_count);
    auto filter_time = omp_get_wtime() - t1;
    std::cout << "Filtering time: " << filter_time << " sec\n";
    std::cout << long_separator;

    // build the candidate search tree
    Edges ***edge_matrix = new Edges **[p.size()];
    for (vidType i = 0; i < p.size(); ++i)
      edge_matrix[i] = new Edges *[p.size()];
    t1 = omp_get_wtime();
    filter.buildTables(&g, &p, candidates, candidates_count, edge_matrix);
    auto build_table_time = omp_get_wtime() - t1;
    size_t mem_cost = filter.computeMemoryCostInBytes(&p, candidates_count, edge_matrix);
    filter.printTableCardinality(&p, edge_matrix);
    std::cout << "Build indices (Candidate Search Tree) time: " << build_table_time << " sec\n";
    std::cout << long_separator;

    // generate query plan
    VertexList matching_order(p.size());
    VertexList pivots(p.size());
    QueryPlan plan("GQL", "LFTJ");
    //QueryPlan plan("GQL", "GQL");
    t1 = omp_get_wtime();
    plan.generate(&g, &p, candidates_count, matching_order, pivots);
    auto plan_gen_time = omp_get_wtime() - t1;
    std::cout << "Generate query plan time: " << plan_gen_time << " sec\n";
    std::cout << long_separator;

    t1 = omp_get_wtime();
    size_t call_count = 0;
    counter = plan.explore(&g, &p, candidates, candidates_count, matching_order, pivots, edge_matrix, call_count);
    auto enumeration_time = omp_get_wtime() - t1;
    std::cout << "Enumeration time: " << enumeration_time << " sec\n";
    printf("Memory cost (MB): %.4lf\n", BYTESTOMB(mem_cost));
    printf("Call Count: %zu\n", call_count);
    printf("Per Call Count Time (seconds): %.4lf\n", enumeration_time / (call_count == 0 ? 1 : call_count));
  } else {
    if (p.is_4color_square()) {
      #include "4-color-square.h"
    } else {
      #include "3-color-triangle.h"
    }
  }
  total = counter;
  t.Stop();
  std::cout << "runtime = " <<  t.Seconds() << " seconds\n";
  return;
}

