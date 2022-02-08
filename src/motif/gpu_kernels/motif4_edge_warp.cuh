
__global__ void motif4_warp_edge(eidType ne, GraphGPU g, vidType *vlists, vidType max_deg, AccType* counters) {
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int num_warps   = WARPS_PER_BLOCK * gridDim.x;            // total number of active warps
  vidType* vlist = &vlists[int64_t(warp_id)*int64_t(max_deg)*2];
  vidType counts[6];
  __shared__ vidType v0[WARPS_PER_BLOCK], v1[WARPS_PER_BLOCK];
  __shared__ vidType v0_size[WARPS_PER_BLOCK], v1_size[WARPS_PER_BLOCK];
  vidType v2, v2_size;
  for (int i = 0; i < 6; i++) counts[i] = 0;
  __shared__ vidType list_size[WARPS_PER_BLOCK][3];
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

    // finding 3-star
    auto cnt = difference_set(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane], v1[warp_lane], vlist);
    if (thread_lane == 0) list_size[warp_lane][0] = cnt;
    __syncwarp();
    for (vidType i = 0; i < list_size[warp_lane][0]; i++) {
      v2 = vlist[i];
      v2_size = g.getOutDegree(v2);
      counts[0] += difference_num(vlist, list_size[warp_lane][0], g.N(v2), v2_size, v2); // 3-star
    }
 
    if (v1[warp_lane] > v0[warp_lane]) continue;

    // finding diamond and tailed_triangle
    cnt = intersect(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane], vlist); // y0y1
    if (thread_lane == 0) list_size[warp_lane][0] = cnt;
    __syncwarp();
    for (vidType j = 0; j < list_size[warp_lane][0]; j++) {
      v2 = vlist[j];
      v2_size = g.getOutDegree(v2);
      counts[4] += difference_num(vlist, list_size[warp_lane][0], g.N(v2), v2_size, v2); // diamond

      cnt = difference_set(g.N(v2), v2_size, v0_ptr, v0_size[warp_lane], &vlist[max_deg]); // n0y2
      if (thread_lane == 0) list_size[warp_lane][1] = cnt;
      __syncwarp();
      counts[2] += difference_num(&vlist[max_deg], list_size[warp_lane][1], v1_ptr, v1_size[warp_lane]); // n0n1y2: tailed_triangle
    }

    // finding 4-clique
    cnt = intersect(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane], v1[warp_lane], vlist); // y0f0y1f1
    if (thread_lane == 0) list_size[warp_lane][0] = cnt;
    __syncwarp();
    for (vidType i = 0; i < list_size[warp_lane][0]; i++) {
      v2 = vlist[i];
      v2_size = g.getOutDegree(v2);
      counts[5] += intersect_num(vlist, list_size[warp_lane][0], g.N(v2), v2_size, v2); // 4-cycle
    }

    // finding 4-path
    cnt = difference_set(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane], vlist); // y0n1
    if (thread_lane == 0) list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v1_ptr, v1_size[warp_lane], v0_ptr, v0_size[warp_lane], &vlist[max_deg]); // n0y1
    if (thread_lane == 0) list_size[warp_lane][1] = cnt;
    __syncwarp();
    for (vidType j = 0; j < list_size[warp_lane][0]; j++) {
      v2 = vlist[j];
      v2_size = g.getOutDegree(v2);
      counts[1] += difference_num(&vlist[max_deg], list_size[warp_lane][1], g.N(v2), v2_size); // 4-path
    }

    // finding 4-cycle
    cnt = difference_set(v1_ptr, v1_size[warp_lane], v0_ptr, v0_size[warp_lane], v0[warp_lane], vlist); // n0f0y1
    if (thread_lane == 0) list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane], v1[warp_lane], &vlist[max_deg]); // y0f0n1f1
    if (thread_lane == 0) list_size[warp_lane][1] = cnt;
    __syncwarp();
    for (vidType j = 0; j < list_size[warp_lane][1]; j++) {
      v2 = vlist[max_deg+j];
      v2_size = g.getOutDegree(v2);
      counts[3] += intersect_num(vlist, list_size[warp_lane][0], g.N(v2), v2_size, v0[warp_lane]); // 4-cycle
    }
  }
  for (int i = 0; i < 6; i++)
    atomicAdd(&counters[i], counts[i]);
}

