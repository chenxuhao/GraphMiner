#include <iostream>
#include "graph.h"
#include "sample.hh"

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: " << argv[0] << " <graph> <path_length> <num_samples>\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph 3 1000\n";
    exit(1);
  }
  Graph g(argv[1]);
  g.print_meta_data();

  int path_length = atoi(argv[2]);
  assert(path_length > 1);
  eidType num_samples = atoi(argv[3]);
  std::cout << "num_samples: " << num_samples << "\n";

  g.init_simple_edgelist();
  auto m = g.E();
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
  random_select_batch<eidType>(0, m, num_samples, edges);
  #pragma omp parallel for reduction(+ : counter) //schedule(dynamic, 1)
  for (eidType i = 0; i < num_samples; i++) {
    uint64_t scale = m;
    auto eid = edges[i];
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    VertexSet vs;
    vs.add(v0);
    vs.add(v1);
    vidType v = 0;
    vidType c = 0;
    for (int j = 1; j < path_length; j++) {
      if (j == path_length-1) {
        c = difference_num(g.N(vs[j]), vs, v0);
      } else {
        VertexSet candidate_set;
        difference_set(candidate_set, g.N(vs[j]), vs);
        c = candidate_set.size();
        auto id = random_select_single(0, c);
        v = candidate_set[id];
        vs.add(v);
      }
      if (c == 0) {
        scale = 0;
        break;
      }
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
