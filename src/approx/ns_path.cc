#include <iostream>
#include <random>
#include "graph.h"
#include <iomanip>
#include <locale>

template<class T>
std::string FormatWithCommas(T value) {
  std::stringstream ss;
  ss.imbue(std::locale(""));
  ss << std::fixed << value;
  return ss.str();
}

vidType sample_neighbor(Graph &g, vidType v, VertexSet &vs, vidType &num, vidType upper_bound = INT_MAX) {
  VertexSet candidate_set;
  if (upper_bound < INT_MAX) 
    difference_set(candidate_set, g.N(v), vs, upper_bound);
  else
    difference_set(candidate_set, g.N(v), vs);
  num = candidate_set.size();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<vidType> dist(0, num);
  auto id = dist(gen);
  return candidate_set[id];
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: " << argv[0] << " <graph> <path_length> <num_samples>\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph 3 1000\n";
    exit(1);
  }
  Graph g(argv[1]);
  g.print_meta_data();
  g.init_simple_edgelist();
  int path_length = atoi(argv[2]);
  assert(path_length > 1);
  eidType num_samples = atoi(argv[3]);
  std::cout << "num_samples: " << num_samples << "\n";
  auto m = g.E();
  Timer t;
  t.Start();
  __int128_t counter = 0;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<eidType> dist(0, m);
  std::vector<eidType> edges(num_samples);
  for (eidType i = 0; i < num_samples; i++)
    edges[i] = dist(gen);

  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";
 
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
    std::vector<vidType> c(path_length-1);
    for (int j = 1; j < path_length; j++) {
      if (j == path_length-1)
        v = sample_neighbor(g, vs[j], vs, c[j-1], v0);
      else
        v = sample_neighbor(g, vs[j], vs, c[j-1]);
      if (c[j-1] == 0) {
        scale = 0;
        break;
      }
      vs.add(v);
      scale *= c[j-1];
    }
    counter += scale;
  }
  // scale down by number of samples
  uint64_t total = counter / num_samples;
  t.Stop();
  std::cout << "runtime = " << t.Seconds() << " sec\n";
  std::cout << "Estimated count " << FormatWithCommas(total) << "\n";
}
