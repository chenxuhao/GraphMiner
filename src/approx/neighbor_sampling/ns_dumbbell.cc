#include "graph.h"
#include "sample.hh"

void sample_dumbbell(Graph &g, eidType num_samples, uint64_t &total);

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <graph> <num_samples>\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph 1000\n";
    exit(1);
  }
  Graph g(argv[1]);
  g.print_meta_data();
  int64_t num_samples = atoi(argv[2]);

  Timer t;
  t.Start();
  uint64_t total = 0;
  sample_dumbbell(g, num_samples, total);
  t.Stop();
  std::cout << "Runtime = " << t.Seconds() << " sec\n";
  std::cout << "Estimated count " << FormatWithCommas(total) << "\n";
}

void sample_dumbbell(Graph &g, eidType num_samples, uint64_t &total) {
  auto m = g.init_edgelist(true);
  std::cout << "m = " << m << "\n";
  double counter = 0;
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";
 
  std::uniform_int_distribution<eidType> edge_dist(0, m-1);
  auto num_samples_per_thread = num_samples / num_threads; 
  int64_t success_sample = 0;

  #pragma omp parallel reduction(+ : counter,success_sample)
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

    VertexSet vs1;
    vs1.add(v1);
    auto y0n1 = difference_set(g.N(v0), vs1);
    auto c0 = y0n1.size();
    if (c0 < 1) continue;
    auto idx0 = random_select_single<vidType>(0, c0-1, gen);
    auto v2 = y0n1[idx0];
    auto y0y2 = intersection_set(y0n1, g.N(v2), v2);
    auto c1 = y0y2.size();
    if (c1 < 1) continue;
    auto idx1 = random_select_single<vidType>(0, c1-1, gen);
    auto v3 = y0y2[idx1];

    VertexSet vs0;
    vs0.add(v0);
    vs0.add(v2);
    vs0.add(v3);
    auto y1n0 = difference_set(g.N(v1), vs0);
    auto c2 = y1n0.size();
    if (c2 < 1) continue;
    auto idx2 = random_select_single<vidType>(0, c2-1, gen);
    auto v4 = y1n0[idx2];
    auto y1y4 = intersection_set(y1n0, g.N(v4), v4);
    auto c3 = y1y4.size();
    if (c3 < 1) continue;
    //auto idx3 = random_select_single<vidType>(0, c3-1, gen);
    //auto v5 = y1y4[idx3];
    //std::cout << "sample succeed!\n";
    success_sample += 1;

    double scale = m;
    counter += scale * c0 * c1 * c2 * c3;
  }
  }
  std::cout << "successful samples = " << success_sample << " sccess_rate = " << double(success_sample) / double(num_samples) << "\n";
  total = counter / num_samples;
}
