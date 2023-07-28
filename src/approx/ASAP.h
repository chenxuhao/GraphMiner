#include "graph.h"

struct {
    vidType v0;
    vidType v1;
    int64_t factor;
} typedef output;

bool get_closure(Graph &g, vidType v1, vidType v2, int depth);

output sample_out_neighbor_edge(Graph &g, vidType v0, vidType v1);

output sample_edge(Graph &g);