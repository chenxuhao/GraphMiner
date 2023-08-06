#include <iostream>
#include "graph.h"
#include "sample.hh"

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: " << argv[0] << " <graph> <k> <num_samples>\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph 3 1000\n";
    exit(1);
  }
  Graph g(argv[1]);
  g.print_meta_data();
  int k = atoi(argv[2]);
  assert(k > 3);
  eidType num_samples = atoi(argv[3]);
  std::cout << "num_samples: " << num_samples << "\n";

  auto m = g.init_edgelist(true);
  std::cout << "num_edges: " << m << "\n";
  __int128_t counter = 0;
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";
 
  Timer t;
  t.Start();
  std::vector<eidType> edges(num_samples);
  random_select_batch<eidType>(0, m-1, num_samples, edges);
  #pragma omp parallel for reduction(+ : counter) //schedule(dynamic, 1)
  for (eidType i = 0; i < num_samples; i++) {
    uint64_t scale = m;
    auto eid = edges[i];
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    VertexSet vs;
    vs.add(v0);
    vs.add(v1);
    vidType v;
    vidType c;
    for (int j = 2; j < k; j++) {
      if (j == k-1) {
        VertexSet candidate_set;
        difference_set(candidate_set, g.N(vs[j-1]), vs, v1);
        scale *= intersection_num(candidate_set, g.N(v0));
        break;
      } else {
        VertexSet candidate_set;
        difference_set(candidate_set, g.N(vs[j-1]), vs, v0);
        c = candidate_set.size();
        auto id = random_select_single(0, c-1);
        v = candidate_set[id];
      }
      if (c == 0) {
        scale = 0;
        break;
      }
      vs.add(v);
      scale *= c;
    }
    counter += scale;
  }
  // scale down by number of samples
  uint64_t total = counter / num_samples;
  t.Stop();
  std::cout << "runtime = " << t.Seconds() << " sec\n";
  std::cout << "Estimated count " << FormatWithCommas(total) << "\n";
}
