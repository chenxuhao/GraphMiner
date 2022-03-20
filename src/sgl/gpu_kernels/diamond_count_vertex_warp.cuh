
__global__ void diamond_warp_vertex_count(vidType nv, GraphGPU g, vidType *vlist, vidType max_deg, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int num_warps   = (BLK_SZ/WARP_SIZE) * gridDim.x;         // total number of active warps
  AccType counter = 0; 
  for (vidType v0 = warp_id; v0 < nv; v0 += num_warps) {
    vidType v0_size = g.getOutDegree(v0);
    for (vidType idx0 = 0; idx0 < v0_size; idx0++) {
      auto v1 = g.N(v0)[idx0]; // v1
      if (v1 >= v0) break; // symmetry breaking
      vidType v1_size = g.getOutDegree(v1);
      AccType count = intersect_num(g.N(v0), v0_size, g.N(v1), v1_size);
      auto n = warp_reduce<AccType>(count);
      if (thread_lane == 0) counter += n * (n-1) / 2;
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

