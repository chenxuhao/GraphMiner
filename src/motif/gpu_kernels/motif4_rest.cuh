__global__ void __launch_bounds__(BLK_SZ, 8)
motif4_rest(eidType ne, GraphGPU g, vidType* vlists, vidType max_deg, AccType *counters) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;       // total number of active warps
  AccType counts[4];
  for (int i = 0; i < 4; i++) counts[i] = 0;
  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    if (v1 > v0) continue;
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    AccType tri = intersect_num(g.N(v0), v0_size, g.N(v1), v1_size); // y0y1
    for (int offset = 16; offset > 0; offset /= 2)
      tri += SHFL_DOWN(tri, offset);
    tri = SHFL(tri, 0);
    if (thread_lane == 0) {
      counts[3] += tri * (tri - 1);
      AccType staru = v0_size - tri - 1;
      AccType starv = v1_size - tri - 1;
      counts[2] += tri * (staru + starv);
      counts[1] += staru * starv;
      counts[0] += staru * (staru - 1);
      counts[0] += starv * (starv - 1);
    }
  }
  atomicAdd(&counters[0], counts[0]);
  atomicAdd(&counters[1], counts[1]);
  atomicAdd(&counters[2], counts[2]);
  atomicAdd(&counters[4], counts[3]);
}

