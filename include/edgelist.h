#pragma once
#include "graph.h"
#ifdef USE_GPU
#include "cutil_subset.h"
#endif

class EdgeList {
  public:
    EdgeList() {}
    EdgeList(Graph &g) { init(g); }
    ~EdgeList() {}
    void init(Graph &g) {
      nnz = g.num_edges();
      src_list.resize(nnz);
      dst_list.resize(nnz);
      uint64_t i = 0;
      for (vidType v = 0; v < g.V(); v ++) {
        for (auto u : g.N(v)) {
          src_list[i] = v;
          dst_list[i] = u;
          i ++;
        }
      }
      assert(i == nnz);
#ifdef USE_GPU
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_src_list, nnz * sizeof(vidType)));
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_dst_list, nnz * sizeof(vidType)));
      CUDA_SAFE_CALL(cudaMemcpy(d_src_list, &src_list[0], nnz * sizeof(vidType), cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(d_dst_list, &dst_list[0], nnz * sizeof(vidType), cudaMemcpyHostToDevice));
#endif
    }
#ifdef USE_GPU
	__device__ vidType get_src(eidType eid) const { return d_src_list[eid]; }
	__device__ vidType get_dst(eidType eid) const { return d_dst_list[eid]; }
#else
	vidType get_src(eidType eid) const { return src_list[eid]; }
	vidType get_dst(eidType eid) const { return dst_list[eid]; }
#endif
	size_t size() { return nnz; }
  private:
    size_t nnz;
    std::vector<vidType> src_list, dst_list;
    vidType *d_src_list, *d_dst_list;
};

