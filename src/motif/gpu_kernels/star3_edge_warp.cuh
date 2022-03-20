// finding 3-star
__global__ void star3_warp_edge(eidType ne, GraphGPU g, vidType *vlists, vidType max_deg, AccType* counters) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int num_warps   = WARPS_PER_BLOCK * gridDim.x;            // total number of active warps
  vidType* vlist = &vlists[int64_t(warp_id)*int64_t(max_deg)];
  __shared__ vidType list_size[WARPS_PER_BLOCK];
  vidType count = 0;
  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    vidType v0 = g.get_src(eid);
    vidType v1 = g.get_dst(eid);
    if (v1 == v0) continue;
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    auto cnt = difference_set(g.N(v0), v0_size, g.N(v1), v1_size, v1, vlist);
    if (thread_lane == 0) list_size[warp_lane] = cnt;
    __syncwarp();
    for (vidType i = 0; i < list_size[warp_lane]; i++) {
      vidType v2 = vlist[i];
      vidType v2_size = g.getOutDegree(v2);
      count += difference_num(vlist, list_size[warp_lane], g.N(v2), v2_size, v2); // 3-star
    }
  }
  //atomicAdd(&counters[0], count);
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) atomicAdd(&counters[0], block_num);
}
/*
__global__ void star3_warp_edge(eidType ne, GraphGPU g, vidType *vlists, vidType max_deg, AccType* counters) {
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int num_warps   = WARPS_PER_BLOCK * gridDim.x;            // total number of active warps
  vidType* vlist = &vlists[int64_t(warp_id)*int64_t(max_deg)];
  __shared__ vidType v0[WARPS_PER_BLOCK], v1[WARPS_PER_BLOCK];
  __shared__ vidType v0_size[WARPS_PER_BLOCK], v1_size[WARPS_PER_BLOCK];
  __shared__ vidType list_size[WARPS_PER_BLOCK];
  vidType count = 0;
  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    if (thread_lane == 0) {
      v0[warp_lane] = g.get_src(eid);
      v1[warp_lane] = g.get_dst(eid);
    }
    __syncwarp();
    if (v1[warp_lane] == v0[warp_lane]) continue;
    if (thread_lane == 0) {
      v0_size[warp_lane] = g.getOutDegree(v0[warp_lane]);
      v1_size[warp_lane] = g.getOutDegree(v1[warp_lane]);
    }
    __syncwarp();
    auto v0_ptr = g.N(v0[warp_lane]);
    auto v1_ptr = g.N(v1[warp_lane]);

    auto cnt = difference_set(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane], v1[warp_lane], vlist);
    if (thread_lane == 0) list_size[warp_lane] = cnt;
    __syncwarp();
    for (vidType i = 0; i < list_size[warp_lane]; i++) {
      vidType v2 = vlist[i];
      vidType v2_size = g.getOutDegree(v2);
      count += difference_num(vlist, list_size[warp_lane], g.N(v2), v2_size, v2); // 3-star
    }
  }
  atomicAdd(&counters[0], count);
}
*/
