#ifndef MINER_HPP_
#define MINER_HPP_
#include "quick_pattern.h"
#include "canonical_graph.h"
#include "types.h"

template <typename T>
inline std::vector<T> parallel_prefix_sum(const std::vector<T> &degrees) {
	std::vector<T> sums(degrees.size() + 1);
	T total = 0;
	for (size_t n = 0; n < degrees.size(); n++) {
		sums[n] = total;
		total += degrees[n];
	}
	sums[degrees.size()] = total;
	return sums;
}

class Miner {
public:
	Miner() {}
	virtual ~Miner() {}
	inline unsigned intersect(unsigned a, unsigned b) {
		return intersect_merge(a, b);
	}
	inline unsigned intersect_dag(unsigned a, unsigned b) {
		return intersect_dag_merge(a, b);
	}

protected:
	Graph *graph;
	std::vector<unsigned> degrees;
	void degree_counting() {
		int m = graph->num_vertices();
		degrees.resize(m);
		for (int i = 0; i < m; i++) {
			degrees[i] = graph->edge_end(i) - graph->edge_begin(i);
		}
	}
	inline unsigned intersect_merge(unsigned src, unsigned dst) {
		unsigned count = 0;
		for (auto e = graph->edge_begin(dst); e != graph->edge_end(dst); e++) {
			auto dst_dst = graph->getEdgeDst(e);
		    for (auto e1 = graph->edge_begin(src); e1 != graph->edge_end(src); e1++) {
				auto to = graph->getEdgeDst(e1);
				if (dst_dst == to) {
					count += 1;
					break;
				}
				if (to > dst_dst) break;
			}
		}
		return count;
	}
	inline unsigned intersect_dag_merge(unsigned p, unsigned q) {
		unsigned count = 0;
		auto p_start = graph->edge_begin(p);
		auto p_end = graph->edge_end(p);
		auto q_start = graph->edge_begin(q);
		auto q_end = graph->edge_end(q);
		auto p_it = p_start;
		auto q_it = q_start;
		int a;
		int b;
		while (p_it < p_end && q_it < q_end) {
			a = graph->getEdgeDst(p_it);
			b = graph->getEdgeDst(q_it);
			int d = a - b;
			if (d <= 0) p_it ++;
			if (d >= 0) q_it ++;
			if (d == 0) count ++;
		}
		return count;
	}
	inline unsigned intersect_search(unsigned a, unsigned b) {
		if (degrees[a] == 0 || degrees[b] == 0) return 0;
		unsigned count = 0;
		unsigned lookup = a;
		unsigned search = b;
		if (degrees[a] > degrees[b]) {
			lookup = b;
			search = a;
		} 
		int begin = graph->edge_begin(search);
		int end = graph->edge_end(search);
		for (auto e = graph->edge_begin(lookup); e != graph->edge_end(lookup); e++) {
			int key = graph->getEdgeDst(e);
			if(binary_search(key, begin, end)) count ++;
		}
		return count;
	}
	inline bool is_all_connected(unsigned dst, const BaseEmbedding &emb, unsigned end, unsigned start = 0) {
		assert(start >= 0 && end > 0);
		bool all_connected = true;
		for(unsigned i = start; i < end; ++i) {
			unsigned from = emb.get_vertex(i);
			if (!is_connected(from, dst)) {
				all_connected = false;
				break;
			}
		}
		return all_connected;
	}
	inline bool is_all_connected_dag(unsigned dst, const BaseEmbedding &emb, unsigned end, unsigned start = 0) {
		assert(start >= 0 && end > 0);
		bool all_connected = true;
		for(unsigned i = start; i < end; ++i) {
			unsigned from = emb.get_vertex(i);
			if (!is_connected_dag(dst, from)) {
				all_connected = false;
				break;
			}
		}
		return all_connected;
	}
	// check if vertex a is connected to vertex b in a undirected graph
	inline bool is_connected(unsigned a, unsigned b) {
		if (degrees[a] == 0 || degrees[b] == 0) return false;
		unsigned key = a;
		unsigned search = b;
		if (degrees[a] < degrees[b]) {
			key = b;
			search = a;
		} 
		auto begin = graph->edge_begin(search);
		auto end = graph->edge_end(search);
		return binary_search(key, begin, end);
	}
	inline bool is_connected_dag(unsigned key, unsigned search) {
		if (degrees[search] == 0) return false;
		auto begin = graph->edge_begin(search);
		auto end = graph->edge_end(search);
		return binary_search(key, begin, end);
	}
	inline bool serial_search(unsigned key, IndexT begin, IndexT end) {
		for (auto offset = begin; offset != end; ++ offset) {
			unsigned d = graph->getEdgeDst(offset);
			if (d == key) return true;
			if (d > key) return false;
		}
		return false;
	}
	inline bool binary_search(unsigned key, IndexT begin, IndexT end) {
		auto l = begin;
		auto r = end-1;
		while (r >= l) { 
			auto mid = l + (r - l) / 2; 
			unsigned value = graph->getEdgeDst(mid);
			if (value == key) return true;
			if (value < key) l = mid + 1; 
			else r = mid - 1; 
		} 
		return false;
	}
	inline void gen_adj_matrix(unsigned n, const std::vector<bool> &connected, Matrix &a) {
		unsigned l = 0;
		for (unsigned i = 1; i < n; i++)
			for (unsigned j = 0; j < i; j++)
				if (connected[l++]) a[i][j] = a[j][i] = 1;
	}
	// calculate the trace of a given n*n matrix
	inline MatType trace(unsigned n, Matrix matrix) {
		MatType tr = 0;
		for (unsigned i = 0; i < n; i++) {
			tr += matrix[i][i];
		}
		return tr;
	}
	// matrix mutiplication, both a and b are n*n matrices
	inline Matrix product(unsigned n, const Matrix &a, const Matrix &b) {
		Matrix c(n, std::vector<MatType>(n));
		for (unsigned i = 0; i < n; ++i) { 
			for (unsigned j = 0; j < n; ++j) { 
				c[i][j] = 0; 
				for(unsigned k = 0; k < n; ++k) {
					c[i][j] += a[i][k] * b[k][j];
				}
			} 
		} 
		return c; 
	}
	// calculate the characteristic polynomial of a n*n matrix A
	inline void char_polynomial(unsigned n, Matrix &A, std::vector<MatType> &c) {
		// n is the size (num_vertices) of a graph
		// A is the adjacency matrix (n*n) of the graph
		Matrix C;
		C = A;
		for (unsigned i = 1; i <= n; i++) {
			if (i > 1) {
				for (unsigned j = 0; j < n; j ++)
					C[j][j] += c[n-i+1];
				C = product(n, A, C);
			}
			c[n-i] -= trace(n, C) / i;
		}
	}
	inline void get_connectivity(unsigned n, unsigned idx, vidType dst, const VertexEmbedding &emb, std::vector<bool> &connected) {
		connected.push_back(true); // 0 and 1 are connected
		for (unsigned i = 2; i < n; i ++)
			for (unsigned j = 0; j < i; j++)
				if (is_connected(emb.get_vertex(i), emb.get_vertex(j)))
					connected.push_back(true);
				else connected.push_back(false);
		for (unsigned j = 0; j < n; j ++) {
			if (j == idx) connected.push_back(true);
			else if (is_connected(emb.get_vertex(j), dst))
				connected.push_back(true);
			else connected.push_back(false);
		}
	}
};

#endif // MINER_HPP_
