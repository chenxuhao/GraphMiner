
// edge parallel: each warp takes one edge
__global__ void diamond_warp_edge(eidType ne, int k, GraphGPU g, vidType *vlist, unsigned max_deg, unsigned long long *total) {
  unsigned thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned warp_id     = thread_id   / WARP_SIZE;                // global warp index
  unsigned thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  unsigned warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  unsigned num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;       // total number of active warps
  unsigned begin = warp_id*max_deg*(k-3);
  unsigned depth = 1;
  Status state = Idle;
  vidType idx[MAX_PATTERN_SIZE-2];
  vidType stack[MAX_PATTERN_SIZE];
  __shared__ vidType vlist_sizes[WARPS_PER_BLOCK][MAX_PATTERN_SIZE-2];
 
  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    if (v1 > v0) continue;
    //if (thread_lane == 0) printf("warp %d v0 %d v1 %d\n", warp_id, v0, v1);
    depth = 1;
    stack[0] = v0;
    stack[1] = v1;
    state = Extending;
    while(1) {
      if (depth == k-2) { // found a match
        unsigned count = 0;
        assert(depth == 2);
        auto u = stack[depth];
        count = count_smaller(u, &vlist[begin+max_deg*(depth-2)], vlist_sizes[warp_lane][depth-2]);
        //if (thread_lane == 0) printf("\t warp %d find %d matches\n", warp_id, count);
        if (thread_lane == 0) atomicAdd(total, count);
        depth --; // backtrack
        state = IteratingEdge;
      } else if (state == Extending) {
        vidType* out_list = &vlist[begin+max_deg*(depth-1)];
        vidType *v_ptr;
        vidType v_size;
        auto u = stack[depth];
        vidType u_size = g.getOutDegree(u);
        if (depth >= 2) {
          v_size = vlist_sizes[warp_lane][depth-2];
          v_ptr = &vlist[begin+max_deg*(depth-2)];
        } else {
          v_size = g.getOutDegree(v0);
          v_ptr = g.N(v0);
        }
        unsigned count = intersect(v_ptr, v_size, g.N(u), u_size, out_list);
        if (thread_lane == 0)
          vlist_sizes[warp_lane][depth-1] = count;
        __syncwarp();
        idx[depth-1] = 0;
      }
      if (depth == 0) break; 
      if (idx[depth-1] == vlist_sizes[warp_lane][depth-1]) {
        if (depth == 1) break; // this subtree is done
        else { // backtrack
          depth --;
          state = IteratingEdge;
        }
      } else {
        auto i = idx[depth-1];
        auto w = vlist[begin+max_deg*(depth-1)+i];
        idx[depth-1] = i + 1;
        depth ++;
        stack[depth] = w;
        state = Extending;
      }
    } // end while
  }
}
