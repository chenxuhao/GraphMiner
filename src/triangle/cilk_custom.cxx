#include "graph.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/opadd_reducer.h>

void TCSolver(Graph &g, uint64_t &total, int, int) {
  int num_threads = __cilkrts_get_nworkers();
  std::cout << "Cilk TC (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  cilk::opadd_reducer<uint64_t> counter = 0;
  #pragma grainsize = 1
  cilk_for (vidType u = 0; u < g.V(); u ++) {
    auto adj_u = g.N(u);
    for (auto v : adj_u) {
    //clik_for (auto i = adj_u.begin(); i != adj_u.end(); i = i+1) {
      //auto v = *i;
      counter += (uint64_t)intersection_num(adj_u, g.N(v));
    }
  }
  total = counter;
  std::cout << "this is the custom cilk file" << "\n";
  t.Stop();
  std::cout << "runtime [cilk_base] = " << t.Seconds() << " sec\n";
  return;
}


