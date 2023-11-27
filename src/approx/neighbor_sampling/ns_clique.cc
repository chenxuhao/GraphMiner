#include "graph.h"
#include "sample.hh"

void sample_clique(Graph &g, int k, eidType num_samples, uint64_t &total);
void sample_4clique(Graph &g, eidType num_samples, uint64_t &counter);

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: " << argv[0] << " <graph> <k> <num_samples>\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph 3 1000\n";
    exit(1);
  }
  Graph g(argv[1], USE_DAG);
  g.print_meta_data();

  int k = atoi(argv[2]);
  assert(k > 2);
  int64_t num_samples = atoi(argv[3]);

  g.init_simple_edgelist();

  Timer t;
  t.Start();
  uint64_t total = 0;
  sample_clique(g, k, num_samples, total);
  t.Stop();
  std::cout << "Runtime = " << t.Seconds() << " sec\n";
  std::cout << "Estimated count " << FormatWithCommas(total) << "\n";
}

void sample_clique(Graph &g, int k, eidType num_samples, uint64_t &total) {
  auto m = g.E();
  double counter = 0;
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";
  std::uniform_int_distribution<eidType> edge_dist(0, m-1);
  auto num_samples_per_thread = num_samples / num_threads; 

  Timer t;
  t.Start();
  #pragma omp parallel reduction(+ : counter)
  {
  std::random_device rd;
  rd_engine gen(rd());
  for (int64_t i = 0; i < num_samples_per_thread; i++) {
    auto eid = edge_dist(gen);
    double scale = m;
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    auto d0 = g.get_degree(v0);
    auto d1 = g.get_degree(v1);
    if (d0 < k-1 || d1 < k-1) continue;
    VertexSet temp[2];
    intersection(g.N(v0), g.N(v1), temp[0]);
    vidType c = temp[0].size();
    if (c == 0) continue;
    if (k == 3) {
      counter += scale * c;
      continue;
    }
    auto idx0 = random_select_single<vidType>(0, c-1, gen);
    vidType v = temp[0][idx0];
    if (g.get_degree(v) < k-1) continue;
    scale *= c;
    for (int j = 2; j < k-1; j++) {
      temp[(j+1)%2].clear();
      if (j == k - 2) {
        c = intersection_num(g.N(v), temp[j%2]);
      } else {
        intersection(g.N(v), temp[j%2], temp[(j+1)%2]);
        c = temp[(j+1)%2].size();
        if (c == 0) { scale = 0; break; }
        auto id = random_select_single<vidType>(0, c-1, gen);
        v = temp[(j+1)%2][id];
      }
      if (g.get_degree(v) < k-1) { scale = 0; break; }
      if (c == 0) { scale = 0; break; }
      scale *= c;
    }
    counter += scale;
  }
  }
  // scale down by number of samples
  total = counter / num_samples;
}

void sample_4clique(Graph &g, eidType num_samples, uint64_t &total) {
  auto m = g.E();
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
    auto y0y1 = g.N(v0) & g.N(v1);
    auto d1 = y0y1.size();
    if (d1 < 1) continue;
    auto idx1 = random_select_single<vidType>(0, d1-1, gen);
    auto v2 = y0y1[idx1];
    counter += intersection_num(y0y1, g.N(v2)) * g.E() * d1;
  }
  }
  total = counter / num_samples;
}
