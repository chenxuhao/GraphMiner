#pragma once
#include "element.cuh"
template <typename ElementTy>
class Embedding {
public:
	Embedding() { size_ = 0; }
	Embedding(size_t n) { size_ = n; elements = new ElementTy[size_]; } // TODO
	//Embedding(const Embedding &emb) { size_ = emb.size(); elements = emb.elements; }
	~Embedding() { }
	__device__ IndexT get_vertex(unsigned i) const { return elements[i].get_vid(); }
	__device__ BYTE get_history(unsigned i) const { return elements[i].get_his(); }
	__device__ BYTE get_label(unsigned i) const { return elements[i].get_vlabel(); }
	__device__ BYTE get_key(unsigned i) const { return elements[i].get_key(); }
	__device__ bool empty() const { return size_ == 0; }
	__device__ size_t size() const { return size_; }
	__device__ ElementTy& back() { return elements[size_-1]; }
	__device__ const ElementTy& back() const { return elements[size_-1]; }
	__device__ ElementTy get_element(unsigned i) const { return elements[i]; }
	__device__ void set_element(unsigned i, ElementTy &ele) { elements[i] = ele; }
	__device__ void set_vertex(unsigned i, IndexT vid) { elements[i].set_vertex_id(vid); }
protected:
	ElementTy *elements;
	size_t size_;
};


class BaseEmbedding : public Embedding<SimpleElement> {
public:
	BaseEmbedding() {}
	BaseEmbedding(size_t n) : Embedding(n) {}
	~BaseEmbedding() {}
};

#ifdef USE_BASE_TYPES
typedef BaseEmbedding EmbeddingType;
#endif

template <typename EmbeddingTy>
class EmbeddingQueue{
public:
	EmbeddingQueue() {}
	~EmbeddingQueue() {}
	void init(int nedges, unsigned max_size = 2, bool use_dag = true) {
		int nnz = nedges;
		if (!use_dag) nnz = nnz / 2;
		size = nedges;
	}
	EmbeddingTy *queue;
	int size;
};

class EmbeddingList {
public:
	EmbeddingList() : last_level(0) {}
	EmbeddingList(int max) : last_level(0) { init_alloc(max); }
	~EmbeddingList() {}
	void init_alloc(int max_size) {
		assert(max_size > 1);
		max_level = max_size;
		h_vid_lists = (IndexT **)malloc(max_level * sizeof(IndexT*));
		h_idx_lists = (IndexT **)malloc(max_level * sizeof(IndexT*));
		CUDA_SAFE_CALL(cudaMalloc(&d_vid_lists, max_level * sizeof(IndexT*)));
		CUDA_SAFE_CALL(cudaMalloc(&d_idx_lists, max_level * sizeof(IndexT*)));
		#ifdef ENABLE_LABEL
		h_his_lists = (BYTE **)malloc(max_level * sizeof(BYTE*));
		CUDA_SAFE_CALL(cudaMalloc(&d_his_lists, max_level * sizeof(BYTE*)));
		#endif
		sizes.resize(max_level);
		sizes[0] = 0;
  }
	void init(eidType nedges, int max_size = 2, bool use_dag = true) {
		init_alloc(max_size);
		int nnz = nedges;
		last_level = 1;
		if (!use_dag) nnz = nnz / 2;
		sizes[1] = nnz;
		CUDA_SAFE_CALL(cudaMalloc((void **)&h_vid_lists[1], nnz * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&h_idx_lists[1], nnz * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMemcpy(d_vid_lists, h_vid_lists, max_level * sizeof(IndexT*), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_idx_lists, h_idx_lists, max_level * sizeof(IndexT*), cudaMemcpyHostToDevice));
		#ifdef ENABLE_LABEL
		CUDA_SAFE_CALL(cudaMalloc((void **)&h_his_lists[1], nnz * sizeof(BYTE)));
		CUDA_SAFE_CALL(cudaMemcpy(d_his_lists, h_his_lists, max_level * sizeof(BYTE*), cudaMemcpyHostToDevice));
		#endif
	}
	void init_cpu(Graph &graph, bool is_dag = false) {
		int nnz = graph.num_edges();
		if (!is_dag) nnz = nnz / 2;
		IndexT *vid_list = (IndexT *)malloc(nnz*sizeof(IndexT));
		IndexT *idx_list = (IndexT *)malloc(nnz*sizeof(IndexT));
		int eid = 0;
		for (int src = 0; src < graph.num_vertices(); src ++) {
			IndexT row_begin = graph.edge_begin(src);
			IndexT row_end = graph.edge_end(src);
			for (IndexT e = row_begin; e < row_end; e++) {
				IndexT dst = graph.getEdgeDst(e);
				if (is_dag || src < dst) {
					vid_list[eid] = dst;
					idx_list[eid] = src;
					eid ++;
				}
			}
		}
		CUDA_SAFE_CALL(cudaMemcpy(h_vid_lists[1], vid_list, nnz * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(h_idx_lists[1], idx_list, nnz * sizeof(IndexT), cudaMemcpyHostToDevice));
		#ifdef ENABLE_LABEL
		CUDA_SAFE_CALL(cudaMemset(h_his_lists[1], 0, nnz * sizeof(BYTE)));
		#endif
	}
	__device__ IndexT get_vid(unsigned level, IndexT id) const { return d_vid_lists[level][id]; }
	__device__ IndexT get_idx(unsigned level, IndexT id) const { return d_idx_lists[level][id]; }
	__device__ BYTE get_his(unsigned level, IndexT id) const { return d_his_lists[level][id]; }
	__device__ unsigned get_pid(IndexT id) const { return pid_list[id]; }
	__device__ void set_vid(unsigned level, IndexT id, IndexT vid) { d_vid_lists[level][id] = vid; }
	__device__ void set_idx(unsigned level, IndexT id, IndexT idx) { d_idx_lists[level][id] = idx; }
	__device__ void set_his(unsigned level, IndexT id, BYTE lab) { d_his_lists[level][id] = lab; }
	__device__ void set_pid(IndexT id, unsigned pid) { pid_list[id] = pid; }
	size_t size() const { return sizes[last_level]; }
	size_t size(unsigned level) const { return sizes[level]; }
	void add_level() {
		last_level ++;
		sizes[last_level] = 0;
		assert(last_level < max_level);
  }
	void add_level(unsigned size) { // TODO: this size could be larger than 2^32, when running LiveJournal and even larger graphs
		last_level ++;
		sizes[last_level] = size;
		assert(last_level < max_level);
		CUDA_SAFE_CALL(cudaMalloc((void **)&h_vid_lists[last_level], size * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&h_idx_lists[last_level], size * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMemcpy(d_vid_lists, h_vid_lists, max_level * sizeof(IndexT*), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_idx_lists, h_idx_lists, max_level * sizeof(IndexT*), cudaMemcpyHostToDevice));
		#ifdef ENABLE_LABEL
		CUDA_SAFE_CALL(cudaMalloc((void **)&h_his_lists[last_level], size * sizeof(BYTE)));
		CUDA_SAFE_CALL(cudaMemcpy(d_his_lists, h_his_lists, max_level * sizeof(BYTE*), cudaMemcpyHostToDevice));
		#endif
		#ifdef USE_PID
		CUDA_SAFE_CALL(cudaMalloc((void **)&pid_list, size * sizeof(unsigned)));
		#endif
	}
	void resize_last_level(unsigned size) {
    resize_level(last_level, size);
  }
	void resize_level(unsigned level, unsigned size) {
		if (sizes[level] >= size) {
      return;
    }
    //std::cout << "resize level " << level << " to be " << size << "\n";
    if (sizes[level] > 0) {
		  CUDA_SAFE_CALL(cudaFree(h_vid_lists[level]));
		  CUDA_SAFE_CALL(cudaFree(h_idx_lists[level]));
		  #ifdef ENABLE_LABEL
		  CUDA_SAFE_CALL(cudaFree(h_his_lists[level]));
      #endif
    }
		CUDA_SAFE_CALL(cudaMalloc((void **)&h_vid_lists[level], size * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&h_idx_lists[level], size * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMemcpy(d_vid_lists, h_vid_lists, max_level * sizeof(IndexT*), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_idx_lists, h_idx_lists, max_level * sizeof(IndexT*), cudaMemcpyHostToDevice));
		#ifdef ENABLE_LABEL
		CUDA_SAFE_CALL(cudaMalloc((void **)&h_his_lists[level], size * sizeof(BYTE)));
		CUDA_SAFE_CALL(cudaMemcpy(d_his_lists, h_his_lists, max_level * sizeof(BYTE*), cudaMemcpyHostToDevice));
		#endif
    sizes[level] = size;
	}

	void remove_tail(unsigned idx) { sizes[last_level] = idx; }
	void reset_level() {
		for (size_t i = 2; i <= last_level; i ++) {
			CUDA_SAFE_CALL(cudaFree(h_vid_lists[i]));
			CUDA_SAFE_CALL(cudaFree(h_idx_lists[i]));
		}
		last_level = 1;
	}
	__device__ void get_embedding(unsigned level, unsigned pos, IndexT *emb) {
		IndexT vid = get_vid(level, pos);
		IndexT idx = get_idx(level, pos);
		emb[level] = vid;
		for (unsigned l = 1; l < level; l ++) {
			vid = get_vid(level-l, idx);
			emb[level-l] = vid;
			idx = get_idx(level-l, idx);
		}
		emb[0] = idx;
	}
	__device__ void get_edge_embedding(unsigned level, unsigned pos, IndexT *vids, BYTE *hiss) {
		IndexT vid = get_vid(level, pos);
		IndexT idx = get_idx(level, pos);
		BYTE his = get_his(level, pos);
		vids[level] = vid;
		hiss[level] = his;
		for (unsigned l = 1; l < level; l ++) {
			vid = get_vid(level-l, idx);
			his = get_his(level-l, idx);
			vids[level-l] = vid;
			hiss[level-l] = his;
			idx = get_idx(level-l, idx);
		}
		vids[0] = idx;
		hiss[0] = 0;
	}

private:
	unsigned max_level;
	unsigned last_level;
	std::vector<size_t> sizes;
	unsigned *pid_list;
	IndexT** h_idx_lists;
	IndexT** h_vid_lists;
	BYTE** h_his_lists;
	IndexT** d_idx_lists;
	IndexT** d_vid_lists;
	BYTE** d_his_lists;
};

__global__ void init_gpu_dag(int m, GraphGPU graph, EmbeddingList emb_list) {
	unsigned src = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < m) {
		IndexT row_begin = graph.edge_begin(src);
		IndexT row_end = graph.edge_end(src);
		for (IndexT e = row_begin; e < row_end; e++) {
			IndexT dst = graph.getEdgeDst(e);
			emb_list.set_vid(1, e, dst);
			emb_list.set_idx(1, e, src);
		}
	}
}

__global__ void init_alloc(int m, GraphGPU graph, EmbeddingList emb_list, IndexT *num_emb) {
	unsigned src = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < m) {
		num_emb[src] = 0;
		#ifdef ENABLE_LABEL
		BYTE src_label = graph.getData(src);
		#endif
		IndexT row_begin = graph.edge_begin(src);
		IndexT row_end = graph.edge_end(src);
		for (IndexT e = row_begin; e < row_end; e++) {
			IndexT dst = graph.getEdgeDst(e);
			#ifdef ENABLE_LABEL
			BYTE dst_label = graph.getData(dst);
			#endif
			#ifdef ENABLE_LABEL
			if (src_label <= dst_label) num_emb[src] ++;
			#else
			if (src < dst) num_emb[src] ++;
			#endif
		}
	}
}

__global__ void init_insert(int m, GraphGPU graph, EmbeddingList emb_list, IndexT *indices) {
	unsigned src = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < m) {
		#ifdef ENABLE_LABEL
		BYTE src_label = graph.getData(src);
		#endif
		IndexT start = indices[src];
		IndexT row_begin = graph.edge_begin(src);
		IndexT row_end = graph.edge_end(src);
		for (IndexT e = row_begin; e < row_end; e++) {
			IndexT dst = graph.getEdgeDst(e);
			#ifdef ENABLE_LABEL
			BYTE dst_label = graph.getData(dst);
			#endif
			#ifdef ENABLE_LABEL
			if (src_label <= dst_label) {
			#else
			if (src < dst) {
			#endif
				emb_list.set_vid(1, start, dst);
				emb_list.set_idx(1, start, src);
				#ifdef ENABLE_LABEL
				emb_list.set_his(1, start, 0);
				#endif
				start ++;
			}
		}
	}
}

