#include "cycle4_edge_warp.cuh"
#include "path4_edge_warp.cuh"
#include "star3_edge_warp.cuh"

__global__ void motif4_triangle(eidType ne, GraphGPU g, vidType *vlists, vidType max_deg, AccType* counters) {
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int num_warps   = WARPS_PER_BLOCK * gridDim.x;            // total number of active warps
  vidType* vlist = &vlists[int64_t(warp_id)*int64_t(max_deg)*2];
  __shared__ vidType v0[WARPS_PER_BLOCK], v1[WARPS_PER_BLOCK];
  __shared__ vidType v0_size[WARPS_PER_BLOCK], v1_size[WARPS_PER_BLOCK];
  __shared__ vidType list_size[WARPS_PER_BLOCK][3];
  vidType counts[3];
  for (int i = 0; i < 3; i++) counts[i] = 0;
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

    // finding diamond and tailed_triangle
    auto cnt = intersect(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane], vlist); // y0y1
    if (thread_lane == 0) list_size[warp_lane][0] = cnt;
    __syncwarp();
    for (vidType j = 0; j < list_size[warp_lane][0]; j++) {
      vidType v2 = vlist[j];
      vidType v2_size = g.getOutDegree(v2);
      counts[1] += difference_num(vlist, list_size[warp_lane][0], g.N(v2), v2_size, v2); // diamond

      auto cnt2 = difference_set(g.N(v2), v2_size, v0_ptr, v0_size[warp_lane], &vlist[max_deg]); // n0y2
      if (thread_lane == 0) list_size[warp_lane][1] = cnt2;
      __syncwarp();
      counts[0] += difference_num(&vlist[max_deg], list_size[warp_lane][1], v1_ptr, v1_size[warp_lane]); // n0n1y2: tailed_triangle
    }

    // finding 4-clique
    for (vidType i = 0; i < list_size[warp_lane][0]; i++) {
      vidType v2 = vlist[i];
      if (v2 > v1[warp_lane]) break;
      vidType v2_size = g.getOutDegree(v2);
      counts[2] += intersect_num(vlist, list_size[warp_lane][0], g.N(v2), v2_size, v2); // 4-cycle
    }
  }
  atomicAdd(&counters[2], counts[0]);
  atomicAdd(&counters[4], counts[1]);
  atomicAdd(&counters[5], counts[2]);
}

__global__ void motif4_wedge(eidType ne, GraphGPU g, vidType *vlists, vidType max_deg, AccType* counters) {
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int num_warps   = WARPS_PER_BLOCK * gridDim.x;            // total number of active warps
  vidType* vlist  = &vlists[int64_t(warp_id)*int64_t(max_deg)*2];
  __shared__ vidType v0[WARPS_PER_BLOCK], v1[WARPS_PER_BLOCK];
  __shared__ vidType v0_size[WARPS_PER_BLOCK], v1_size[WARPS_PER_BLOCK];
  __shared__ vidType list_size[WARPS_PER_BLOCK][3];
  vidType counts[2];
  for (int i = 0; i < 2; i++) counts[i] = 0;
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

    // finding 4-path
    auto cnt = difference_set(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane], vlist); // y0n1
    if (thread_lane == 0) list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v1_ptr, v1_size[warp_lane], v0_ptr, v0_size[warp_lane], &vlist[max_deg]); // n0y1
    if (thread_lane == 0) list_size[warp_lane][1] = cnt;
    __syncwarp();
    for (vidType j = 0; j < list_size[warp_lane][0]; j++) {
      vidType v2 = vlist[j];
      vidType v2_size = g.getOutDegree(v2);
      counts[0] += difference_num(&vlist[max_deg], list_size[warp_lane][1], g.N(v2), v2_size); // 4-path
    }

    // finding 4-cycle
    cnt = difference_set(v1_ptr, v1_size[warp_lane], v0_ptr, v0_size[warp_lane], v0[warp_lane], vlist); // n0f0y1
    if (thread_lane == 0) list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane], v1[warp_lane], &vlist[max_deg]); // y0f0n1f1
    if (thread_lane == 0) list_size[warp_lane][1] = cnt;
    __syncwarp();
    for (vidType j = 0; j < list_size[warp_lane][1]; j++) {
      vidType v2 = vlist[max_deg+j];
      vidType v2_size = g.getOutDegree(v2);
      counts[1] += intersect_num(vlist, list_size[warp_lane][0], g.N(v2), v2_size, v0[warp_lane]); // 4-cycle
    }
  }
  atomicAdd(&counters[1], counts[0]);
  atomicAdd(&counters[3], counts[1]);
}

