// warp-centric vertex-parallel: each warp takes one vertex
__global__ void __launch_bounds__(BLK_SZ, 8)
diamond_warp_vertex_nested(vidType nv, GraphGPU g, vidType *vlist, vidType max_deg, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  unsigned thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned warp_id     = thread_id   / WARP_SIZE;                // global warp index
  unsigned thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  unsigned warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  unsigned num_warps   = (BLK_SZ/WARP_SIZE) * gridDim.x;         // total number of active warps
  __shared__ vidType vlist_sizes[BLK_SZ/WARP_SIZE];
  vidType begin = warp_id*max_deg;
  AccType counter = 0; 
  for (vidType v0 = warp_id; v0 < nv; v0 += num_warps) {
    vidType v0_size = g.getOutDegree(v0);
    for (vidType idx0 = 0; idx0 < v0_size; idx0++) {
      auto v1 = g.N(v0)[idx0]; // v1
      if (v1 >= v0) break; // symmetry breaking
      vidType v1_size = g.getOutDegree(v1);
      auto count = intersect(g.N(v0), v0_size, g.N(v1), v1_size, &vlist[begin]);
      if (thread_lane == 0) vlist_sizes[warp_lane] = count;
      __syncwarp();
      for (vidType i = 0; i < vlist_sizes[warp_lane]; i++) {
        vidType v2 = vlist[begin+i]; // v2
        AccType count3 = count_smaller(v2, &vlist[begin], vlist_sizes[warp_lane]);
        if (thread_lane == 0) counter += count3;
      }
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

