// edge-parallel warp-centric: each warp takes one edge
// this kernel is for the DAG version: no need to do symmetry breaking on-the-fly
__global__ void __launch_bounds__(BLK_SZ, 8)
clique5_warp_edge(eidType ne, GraphGPU g, vidType *vlists, vidType max_deg, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;       // total number of active warps
  vidType *vlist  = &vlists[int64_t(warp_id)*int64_t(max_deg)*2];
  AccType counter = 0;
  __shared__ vidType list_size[WARPS_PER_BLOCK][2];
  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    auto v0_size = g.get_degree(v0);
    auto v1_size = g.get_degree(v1);
    auto count1 = intersect(g.N(v0), v0_size, g.N(v1), v1_size, vlist);
    if (thread_lane == 0) list_size[warp_lane][0] = count1;
    __syncwarp();
    for (vidType i = 0; i < list_size[warp_lane][0]; i++) {
      auto v2 = vlist[i];
      auto v2_size = g.get_degree(v2);
      auto w1_size = list_size[warp_lane][0];
      auto count2 = intersect(vlist, w1_size, g.N(v2), v2_size, vlist+max_deg);
      if (thread_lane == 0) list_size[warp_lane][1] = count2;
      __syncwarp();
      for (vidType j = 0; j < list_size[warp_lane][1]; j++) {
        auto v3 = vlist[max_deg+j];
        auto v3_size = g.get_degree(v3);
        auto w2_size = list_size[warp_lane][1];
        counter += intersect_num(vlist+max_deg, w2_size, g.N(v3), v3_size);
      }
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

