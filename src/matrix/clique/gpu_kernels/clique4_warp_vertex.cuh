// edge-parallel warp-centric: each warp takes one edge
// this kernel is for the DAG version: no need to do symmetry breaking on-the-fly
__global__ void __launch_bounds__(BLK_SZ, 8)
clique4_warp_vertex(vidType nv, GraphGPU g, vidType *vlists, vidType max_deg, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id / WARP_SIZE;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  int warp_lane   = threadIdx.x / WARP_SIZE;
  int num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;
  vidType *vlist  = &vlists[int64_t(warp_id)*int64_t(max_deg)];
  AccType counter = 0;
  //__shared__ vidType list_size[WARPS_PER_BLOCK];
  for (auto v0 = warp_id; v0 < nv; v0 += num_warps) {
    auto v0_ptr = g.N(v0);
    auto v0_size = g.get_degree(v0);
    for (vidType i = 0; i < v0_size; i++) {
      auto v1 = v0_ptr[i];
      auto v1_size = g.get_degree(v1);
      auto count = intersect(v0_ptr, v0_size, g.N(v1), v1_size, vlist);
      __syncwarp();
      for (vidType i = 0; i < count; i++) {
        auto u = vlist[i];
        auto u_size = g.get_degree(u);
        //auto v_size = list_size[warp_lane];
        counter += intersect_num(vlist, count, g.N(u), u_size);
      }
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

