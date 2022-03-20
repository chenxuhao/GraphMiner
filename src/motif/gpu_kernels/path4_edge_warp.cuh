// finding 4-path
__global__ void path4_warp_edge(eidType ne, GraphGPU g, vidType *vlists, vidType max_deg, AccType* counters) {
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int num_warps   = WARPS_PER_BLOCK * gridDim.x;            // total number of active warps
  vidType* vlist  = &vlists[int64_t(warp_id)*int64_t(max_deg)*2];
  __shared__ vidType list_size[WARPS_PER_BLOCK][2];
  vidType count = 0;
  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    vidType v0 = g.get_src(eid);
    vidType v1 = g.get_dst(eid);
    if (v1 >= v0) continue;
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    auto v0_ptr = g.N(v0);
    auto v1_ptr = g.N(v1);
    auto cnt = difference_set(v0_ptr, v0_size, v1_ptr, v1_size, vlist); // y0n1
    if (thread_lane == 0) list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v1_ptr, v1_size, v0_ptr, v0_size, &vlist[max_deg]); // n0y1
    if (thread_lane == 0) list_size[warp_lane][1] = cnt;
    __syncwarp();
    for (vidType j = 0; j < list_size[warp_lane][0]; j++) {
      vidType v2 = vlist[j];
      vidType v2_size = g.getOutDegree(v2);
      count += difference_num(&vlist[max_deg], list_size[warp_lane][1], g.N(v2), v2_size); // 4-path
    }
  }
  atomicAdd(&counters[1], count);
}
/*
__global__ void path4_warp_edge(eidType ne, GraphGPU g, vidType *vlists, vidType max_deg, AccType* counters) {
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int num_warps   = WARPS_PER_BLOCK * gridDim.x;            // total number of active warps
  vidType* vlist  = &vlists[int64_t(warp_id)*int64_t(max_deg)*2];
  __shared__ vidType v0[WARPS_PER_BLOCK], v1[WARPS_PER_BLOCK];
  __shared__ vidType v0_size[WARPS_PER_BLOCK], v1_size[WARPS_PER_BLOCK];
  __shared__ vidType list_size[WARPS_PER_BLOCK][3];
  vidType count = 0;
  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    if (thread_lane == 0) {
      v0[warp_lane] = g.get_src(eid);
      v1[warp_lane] = g.get_dst(eid);
    }
    __syncwarp();
    if (v1[warp_lane] >= v0[warp_lane]) continue;
    if (thread_lane == 0) {
      v0_size[warp_lane] = g.getOutDegree(v0[warp_lane]);
      v1_size[warp_lane] = g.getOutDegree(v1[warp_lane]);
    }
    __syncwarp();
    auto v0_ptr = g.N(v0[warp_lane]);
    auto v1_ptr = g.N(v1[warp_lane]);
    auto cnt = difference_set(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane], vlist); // y0n1
    if (thread_lane == 0) list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v1_ptr, v1_size[warp_lane], v0_ptr, v0_size[warp_lane], &vlist[max_deg]); // n0y1
    if (thread_lane == 0) list_size[warp_lane][1] = cnt;
    __syncwarp();
    for (vidType j = 0; j < list_size[warp_lane][0]; j++) {
      vidType v2 = vlist[j];
      vidType v2_size = g.getOutDegree(v2);
      count += difference_num(&vlist[max_deg], list_size[warp_lane][1], g.N(v2), v2_size); // 4-path
    }
  }
  atomicAdd(&counters[1], count);
}
*/
