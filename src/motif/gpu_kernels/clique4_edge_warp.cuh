// edge-parallel warp-centric: each warp takes one edge
__global__ void __launch_bounds__(BLK_SZ, 8)
clique4_warp_edge(eidType ne, GraphGPU g, vidType *vlists, vidType max_deg, AccType *counters) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;       // total number of active warps
  vidType* vlist  = &vlists[int64_t(warp_id)*int64_t(max_deg)];
  AccType counts = 0;
  __shared__ vidType list_size[WARPS_PER_BLOCK];
  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    if (v1 > v0) continue;
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    auto count = intersect(g.N(v0), v0_size, g.N(v1), v1_size, v1, vlist); // y0f0y1f1
    if (thread_lane == 0) list_size[warp_lane] = count;
    __syncwarp();
    for (vidType i = 0; i < list_size[warp_lane]; i++) {
      vidType v2 = vlist[i];
      vidType v2_size = g.getOutDegree(v2);
      vidType v_size = list_size[warp_lane];
      counts += intersect_num(vlist, v_size, g.N(v2), v2_size, v2);
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counts);
  if (threadIdx.x == 0) atomicAdd(&counters[5], block_num);
}

