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
  std::cout << "num_samples: " << num_samples << "\n";

  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";

  Timer t;
  t.Start();
  __int128_t counter = 0;
  std::vector<eidType> edges(num_samples);
  random_select_batch<eidType>(0, m-1, num_samples, edges);
  #pragma omp parallel for reduction(+ : counter) //schedule(dynamic, 1)
  for (int64_t i = 0; i < num_samples; i++) {
    uint64_t scale = m;
    auto eid = edges[i];
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    VertexSet vs;
    VertexSet temp[2];
    vs.add(v0);
    vs.add(v1);
    vidType v;

    intersection(g.N(v0), g.N(v1), temp[0]);
    vidType c = temp[0].size();
    if (c == 0) continue;
    if (k == 3) {
      counter += scale * c;
      continue;
    }
    auto idx0 = random_select_single<vidType>(0, c-1);
    v = temp[0][idx0];
    vs.add(v);
    scale *= c;
    for (int j = 2; j < k-1; j++) {
      temp[(j+1)%2].clear();
      if (j == k - 2)
        c = intersection_num(g.N(vs[j]), temp[j%2]);
      else {
        intersection(g.N(vs[j]), temp[j%2], temp[(j+1)%2]);
        c = temp[(j+1)%2].size();
        if (c == 0) { scale = 0; break; }
        auto id = random_select_single<vidType>(0, c-1);
        v = temp[(j+1)%2][id];
        vs.add(v);
      }
      if (c == 0) { scale = 0; break; }
      scale *= c;
    }
    counter += scale;
  }
  // scale down by number of samples
  total = counter / num_samples;
}

void sample_4clique(Graph &g, eidType num_samples, uint64_t &total) {
  auto m = g.E();
  std::vector<eidType> edges(num_samples);
  random_select_batch<eidType>(0, m-1, num_samples, edges);
  __int128_t counter = 0;
  #pragma omp parallel for reduction(+ : counter) //schedule(dynamic, 1)
  for (eidType i = 0; i < num_samples; i++) {
    auto eid = edges[i];
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    auto y0y1 = g.N(v0) & g.N(v1);
    auto d1 = y0y1.size();
    if (d1 == 0) continue;
    auto idx1 = random_select_single<vidType>(0, d1-1);
    auto v2 = y0y1[idx1];
    counter += intersection_num(y0y1, g.N(v2)) * g.E() * d1;
  }
  total = counter / num_samples;
}
