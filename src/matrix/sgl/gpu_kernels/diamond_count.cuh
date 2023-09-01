// edge parallel: each warp takes one edge
__global__ void __launch_bounds__(BLK_SZ, 8)
diamond_warp_edge_count(eidType ne, GraphGPU g, vidType *vlist, vidType max_deg, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int num_warps   = (BLK_SZ/WARP_SIZE) * gridDim.x;         // total number of active warps
  AccType counter = 0; 
  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    AccType count = intersect_num(g.N(v0), v0_size, g.N(v1), v1_size);
    auto n = warp_reduce<AccType>(count);
    if (thread_lane == 0) counter += n * (n-1) / 2;
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

