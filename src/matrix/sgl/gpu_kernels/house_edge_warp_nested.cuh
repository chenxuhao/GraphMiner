
// edge parallel: each warp takes one edge
__global__ void __launch_bounds__(BLK_SZ, 6)
house_warp_edge_nested(eidType ne, GraphGPU g, vidType *vlist, vidType max_deg, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;       // total number of active warps
  __shared__ vidType vlist_sizes[WARPS_PER_BLOCK];
  vidType ancestors[MAX_PATTERN_SIZE];
  AccType counter = 0; 
  unsigned begin = warp_id*max_deg;
  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    unsigned count = intersect(g.N(v0), v0_size, g.N(v1), v1_size, &vlist[begin]);
    if (thread_lane == 0) vlist_sizes[warp_lane] = count;
    __syncwarp();
    for (vidType i = 0; i < vlist_sizes[warp_lane]; i++) {
      vidType v2 = vlist[begin+i];
      //vidType v2_size = g.getOutDegree(v2);
      for (vidType idx3 = 0; idx3 < v1_size; idx3++) {
        vidType v3 = g.N(v1)[idx3];
        if (v3 == v0 || v3 == v2) continue;
        vidType v3_size = g.getOutDegree(v3);
        ancestors[0] = v1;
        ancestors[1] = v2;
        counter += intersect_num(g.N(v0), v0_size, g.N(v3), v3_size, ancestors, 2); // v4 != v1 && v4 != v2 
      }
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}
