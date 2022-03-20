// edge parallel: each warp takes one edge
__global__ void __launch_bounds__(BLK_SZ, 8)
motif3_formula_warp_vertex(vidType nv, GraphGPU g, AccType *counters) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x; // global thread id
  int warp_id     = thread_id   / WARP_SIZE;               // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);           // thread index within the warp
  int num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;      // total number of active warps
  AccType tri_count = 0;
  AccType wed_count = 0;
  for (vidType v0 = warp_id; v0 < nv; v0 += num_warps) {
    vidType v0_size = g.getOutDegree(v0);
    AccType count = AccType(v0_size) * AccType(v0_size-1);
    if (thread_lane == 0) wed_count += count;
    auto begin = g.edge_begin(v0);
    auto end = g.edge_end(v0);
    for (auto e = begin; e < end; e++) {
      auto v1 = g.getEdgeDst(e);
      vidType v1_size = g.getOutDegree(v1);
      if (v1 >= v0) break;
      tri_count += intersect_num(g.N(v0), v0_size, g.N(v1), v1_size, v1);
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(tri_count);
  if (threadIdx.x == 0) atomicAdd(&counters[0], block_num);
  block_num = BlockReduce(temp_storage).Sum(wed_count);
  if (threadIdx.x == 0) atomicAdd(&counters[1], block_num);
}

