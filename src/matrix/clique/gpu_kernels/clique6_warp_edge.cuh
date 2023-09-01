// edge-parallel warp-centric: each warp takes one edge
// this kernel is for the DAG version: no need to do symmetry breaking on-the-fly
__global__ void __launch_bounds__(BLK_SZ, 6)
clique6_warp_edge(eidType ne, GraphGPU g, vidType *vlists, vidType max_deg, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;       // total number of active warps
  vidType *vlist  = &vlists[int64_t(warp_id)*int64_t(max_deg)*3];
  AccType counter = 0;
  __shared__ vidType list_size[WARPS_PER_BLOCK][3];
  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    auto count1 = intersect(g.N(v0), v0_size, g.N(v1), v1_size, vlist);
    if (thread_lane == 0) list_size[warp_lane][0] = count1;
    __syncwarp();
    for (vidType idx1 = 0; idx1 < list_size[warp_lane][0]; idx1++) {
      vidType v2 = vlist[idx1];
      vidType v2_size = g.getOutDegree(v2);
      vidType w1_size = list_size[warp_lane][0];
      auto count2 = intersect(vlist, w1_size, g.N(v2), v2_size, vlist+max_deg);
      if (thread_lane == 0) list_size[warp_lane][1] = count2;
      __syncwarp();
      for (vidType idx2 = 0; idx2 < list_size[warp_lane][1]; idx2++) {
        vidType v3 = vlist[max_deg+idx2];
        vidType v3_size = g.getOutDegree(v3);
        vidType w2_size = list_size[warp_lane][1];
        auto count3 = intersect(vlist+max_deg, w2_size, g.N(v3), v3_size, vlist+max_deg*2);
        if (thread_lane == 0) list_size[warp_lane][2] = count3;
        __syncwarp();
        for (vidType idx3 = 0; idx3 < list_size[warp_lane][2]; idx3++) {
          vidType v4 = vlist[max_deg*2+idx3];
          vidType v4_size = g.getOutDegree(v4);
          vidType w3_size = list_size[warp_lane][2];
          counter += intersect_num(vlist+max_deg*2, w3_size, g.N(v4), v4_size);
        }
      }
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

