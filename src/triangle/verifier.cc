// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "tc.h"
#include "timer.h"

void TCVerifier(Graph &g, uint64_t test_total) {
	printf("Verifying...\n");
	uint64_t total = 0;
	Timer t;
	t.Start();
	for (VertexId u = 0; u < g.V(); u ++) {
    auto yu = g.N(u);
    for (auto v : yu) {
      total += (uint64_t)intersection_num(yu, g.N(v));
    }
	}
	t.Stop();
	printf("\truntime [serial] = %f sec\n", t.Seconds());
	if (total == test_total) printf("Correct\n");
	else printf("Wrong\n");
	std::cout << "total " << total << " test_total " << test_total << std::endl;
	return;
}

