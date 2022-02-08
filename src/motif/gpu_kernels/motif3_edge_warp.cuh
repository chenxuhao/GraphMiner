// edge parallel: each warp takes one edge
__global__ void motif3_warp_edge(eidType ne, GraphGPU g, vidType *vlist, vidType max_deg, AccType *counters) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread id
  int warp_id   = thread_id / WARP_SIZE;                 // global warp index
  int num_warps = WARPS_PER_BLOCK * gridDim.x;           // total number of active warps
  AccType tri_count = 0;
  AccType wed_count = 0;
  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    if (v1 == v0) continue;
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    wed_count += difference_num(g.N(v0), v0_size, g.N(v1), v1_size, v1);
    if (v1 >= v0) continue;
    tri_count += intersect_num(g.N(v0), v0_size, g.N(v1), v1_size, v1);
  }
  AccType block_num = BlockReduce(temp_storage).Sum(tri_count);
  if (threadIdx.x == 0) atomicAdd(&counters[0], block_num);
  block_num = BlockReduce(temp_storage).Sum(wed_count);
  if (threadIdx.x == 0) atomicAdd(&counters[1], block_num);
}

