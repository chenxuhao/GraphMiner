// vertex paralle: each warp takes one vertex
__global__ void warp_vertex(int nv, int k, GraphGPU g, vidType *vlist, unsigned max_deg, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  unsigned thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  unsigned warp_id     = thread_id   / WARP_SIZE;                // global warp index
  unsigned warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  unsigned num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;   // total number of active warps

  unsigned begin = warp_id*max_deg*(k-3);
  unsigned depth = 1;
  Status state = Idle;
  vidType idx[MAX_PATTERN_SIZE-2];
  vidType stack[MAX_PATTERN_SIZE];
  AccType local_num = 0;
  __shared__ vidType vlist_sizes[WARPS_PER_BLOCK][MAX_PATTERN_SIZE-2];
 
  AccType counter = 0;
  for (auto v0 = warp_id; v0 < nv; v0 += num_warps) {
    stack[0] = v0;
    for (auto e = g.edge_begin(v0); e < g.edge_end(v0); e ++) {
      auto v1 = g.getEdgeDst(e);
      depth = 1;
      stack[1] = v1;
      state = Extending;
      while(1) {
        if (depth == k-2) { // found a match
          auto u = stack[depth];
          vidType u_size = g.getOutDegree(u);
          if (depth >= 2) {
            local_num += intersect_num(&vlist[begin+max_deg*(depth-2)], vlist_sizes[warp_lane][depth-2], g.N(u), u_size);
          } else {
            assert(depth == 1);
            vidType v0_size = g.getOutDegree(v0);
            local_num += intersect_num(g.N(v0), v0_size, g.N(u), u_size);
          }
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
          assert(count <= max_deg);
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
    } // end v1
  } // end v0
  AccType block_num = BlockReduce(temp_storage).Sum(local_num);
  if(threadIdx.x == 0) atomicAdd(total, block_num);
}

