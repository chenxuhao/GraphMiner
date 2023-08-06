#include "graph.h"
#include "sample.hh"

void sample_house(Graph &g, eidType num_samples, uint64_t &total);

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <graph> <num_samples>\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph 3 1000\n";
    exit(1);
  }
  Graph g(argv[1]);
  g.print_meta_data();
  int64_t num_samples = atoi(argv[2]);

  Timer t;
  t.Start();
  uint64_t total = 0;
  sample_house(g, num_samples, total);
  t.Stop();
  std::cout << "Runtime = " << t.Seconds() << " sec\n";
  std::cout << "Estimated count " << FormatWithCommas(total) << "\n";
}

void sample_house(Graph &g, eidType num_samples, uint64_t &total) {
  auto m = g.init_edgelist(true);
  double counter = 0;
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";
 
  std::uniform_int_distribution<eidType> edge_dist(0, m-1);
  auto num_samples_per_thread = num_samples / num_threads; 

  #pragma omp parallel reduction(+ : counter)
  {
  std::random_device rd;
  rd_engine gen(rd());
  for (int64_t i = 0; i < num_samples_per_thread; i++) {
    auto eid = edge_dist(gen);
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    auto d0 = g.get_degree(v0);
    if (d0 < 3) continue;
    auto d1 = g.get_degree(v1);
    if (d1 < 3) continue;
    auto y0y1 = g.N(v0) & g.N(v1);
    auto c0 = y0y1.size();
    if (c0 < 1) continue;
    auto idx0 = random_select_single<vidType>(0, c0-1, gen);
    auto v2 = y0y1[idx0];
    auto u = v1 > v2 ? v2 : v1;
    auto w = v1 > v2 ? v1 : v2;
    VertexSet vs12;
    vs12.add(u);
    vs12.add(w);
    auto candidate_v3 = difference_set(g.N(v0), vs12);
    auto c1 = candidate_v3.size();
    if (c1 < 1) continue;
    auto idx1 = random_select_single<vidType>(0, c1-1, gen);
    auto v3 = candidate_v3[idx1];
    u = v0 > v2 ? v2 : v0;
    w = v0 > v2 ? v0 : v2;
    VertexSet vs02;
    vs02.add(u);
    vs02.add(w);
    auto candidate_v4 = difference_set(g.N(v1), vs02);
    auto c2 = intersection_num(g.N(v3), candidate_v4);
    if (c2 < 1) continue;
    double scale = m;
    counter += scale * c0 * c1 * c2;
  }
  }
  total = counter / num_samples;
}
