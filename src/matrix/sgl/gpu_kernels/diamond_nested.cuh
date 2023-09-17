// Copyright 2020 MIT
// Contact: Xuhao Chen <cxh@mit.edu>
__global__ void __launch_bounds__(BLK_SZ, 8)
diamond_warp_edge_nested(eidType ne, GraphGPU g, vidType *vlist, vidType max_deg, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int num_warps   = (BLK_SZ/WARP_SIZE) * gridDim.x;         // total number of active warps
  __shared__ vidType vlist_sizes[BLK_SZ/WARP_SIZE];
  vidType v[3];
  AccType counter = 0; 
  vidType begin = warp_id*max_deg;
  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    v[0] = g.get_src(eid);
    v[1] = g.get_dst(eid);
    vidType v0_size = g.getOutDegree(v[0]);
    vidType v1_size = g.getOutDegree(v[1]);
    auto count = intersect(g.N(v[0]), v0_size, g.N(v[1]), v1_size, &vlist[begin]);
    if (thread_lane == 0) vlist_sizes[warp_lane] = count;
    __syncwarp();
    for (vidType i = 0; i < vlist_sizes[warp_lane]; i++) {
      v[2] = vlist[begin+i];
      AccType count3 = count_smaller(v[2], &vlist[begin], vlist_sizes[warp_lane]);
      if (thread_lane == 0) counter += count3;
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

