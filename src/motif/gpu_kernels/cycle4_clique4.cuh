
__global__ void cycle4_clique4(eidType ne, GraphGPU g, vidType* vlists, vidType max_deg, AccType *counters) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;       // total number of active warps
  vidType* vlist = &vlists[int64_t(warp_id)*int64_t(max_deg)*2];
  AccType counts[2];
  for (int i = 0; i < 2; i++) counts[i] = 0;
  __shared__ vidType list_size[WARPS_PER_BLOCK][2];
  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    if (v1 > v0) continue;
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);

    // finding 4-clique
    unsigned count1 = intersect(g.N(v0), v0_size, g.N(v1), v1_size, v1, vlist); // y0f0y1f1
    if (thread_lane == 0) list_size[warp_lane][0] = count1;
    __syncwarp();
    for (unsigned i = 0; i < list_size[warp_lane][0]; i++) {
      vidType v2 = vlist[i];
      vidType v2_size = g.getOutDegree(v2);
      counts[0] += intersect_num(vlist, list_size[warp_lane][0], g.N(v2), v2_size, v2); // 4-clique
    }

    // finding 4-cycle
    count1 = difference_set(g.N(v1), v1_size, g.N(v0), v0_size, v0, vlist); // n0f0y1
    if (thread_lane == 0) list_size[warp_lane][0] = count1;
    __syncwarp();
    unsigned count2 = difference_set(g.N(v0), v0_size, g.N(v1), v1_size, v1, &vlist[max_deg]); // y0f0n1f1
    if (thread_lane == 0) list_size[warp_lane][1] = count2;
    __syncwarp();
    for (unsigned j = 0; j < list_size[warp_lane][1]; j++) {
      vidType v2 = vlist[max_deg+j];
      vidType v2_size = g.getOutDegree(v2);
      counts[1] += intersect_num(vlist, list_size[warp_lane][0], g.N(v2), v2_size, v0); // 4-cycle
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counts[0]);
  if (threadIdx.x == 0) atomicAdd(&counters[5], block_num);
  block_num = BlockReduce(temp_storage).Sum(counts[1]);
  if (threadIdx.x == 0) atomicAdd(&counters[3], block_num);
}

