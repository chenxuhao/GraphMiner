// vertex paralle: each warp takes one vertex
__global__ void warp_vertex(vidType begin, vidType end, GraphGPU g, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id     = thread_id   / WARP_SIZE;               // global warp index
  int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  AccType count = 0;
  for (auto v = warp_id+begin; v < end; v += num_warps) {
    vidType *v_ptr = g.N(v);
    vidType v_size = g.getOutDegree(v);
    for (auto e = 0; e < v_size; e ++) {
      auto u = v_ptr[e];
      vidType u_size = g.getOutDegree(u);
      count += intersect_num(v_ptr, v_size, g.N(u), u_size);
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if(threadIdx.x == 0) atomicAdd(total, block_num);
}

