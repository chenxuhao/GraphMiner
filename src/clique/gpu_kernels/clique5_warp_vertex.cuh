// edge-parallel warp-centric: each warp takes one edge
// this kernel is for the DAG version: no need to do symmetry breaking on-the-fly
__global__ void __launch_bounds__(BLK_SZ, 8)
clique5_warp_vertex(vidType nv, GraphGPU g, vidType *vlists, vidType max_deg, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id / WARP_SIZE;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  int warp_lane   = threadIdx.x / WARP_SIZE;
  int num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;
  vidType *vlist  = &vlists[int64_t(warp_id)*int64_t(max_deg)*2];
  AccType counter = 0;
  //__shared__ vidType list_size[WARPS_PER_BLOCK][2];
  for (auto v0 = warp_id; v0 < nv; v0 += num_warps) {
    auto v0_size = g.get_degree(v0);
    auto v0_ptr = g.N(v0);
    for (vidType i0 = 0; i0 < v0_size; i0++) {
      auto v1 = v0_ptr[i0];
      auto v1_size = g.get_degree(v1);
      auto count1 = intersect(v0_ptr, v0_size, g.N(v1), v1_size, vlist);
      for (vidType i1 = 0; i1 < count1; i1++) {
        auto v2 = vlist[i1];
        auto v2_size = g.get_degree(v2);
        auto count2 = intersect(vlist, count1, g.N(v2), v2_size, vlist+max_deg);
        for (vidType i2 = 0; i2 < count2; i2++) {
          auto v3 = vlist[max_deg+i2];
          auto v3_size = g.get_degree(v3);
          counter += intersect_num(vlist+max_deg, count2, g.N(v3), v3_size);
        }
      }
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

