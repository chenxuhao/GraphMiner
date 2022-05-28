#ifndef EDGE_MINER_H
#define EDGE_MINER_H
#include <mutex>
#include <numeric>
#include "miner.h"
#include "domain_support.h"

typedef std::pair<unsigned, unsigned> InitPattern;
typedef QuickPattern<EdgeEmbedding, ElementType> QPattern;
typedef CanonicalGraph<EdgeEmbedding, ElementType> CPattern;
typedef std::unordered_map<QPattern, Frequency> QpMapFreq; // quick pattern map (mapping quick pattern to its frequency)
typedef std::unordered_map<CPattern, Frequency> CgMapFreq; // canonical pattern map (mapping canonical pattern to its frequency)
typedef std::map<InitPattern, DomainSupport*> InitMap;
typedef std::unordered_map<QPattern, DomainSupport*> QpMapDomain; // quick pattern map (mapping quick pattern to its domain support)
typedef std::unordered_map<CPattern, DomainSupport*> CgMapDomain; // canonical pattern map (mapping canonical pattern to its domain support)
typedef std::unordered_map<unsigned, unsigned> FreqMap;
typedef std::unordered_map<unsigned, bool> DomainMap;
typedef PerThreadStorage<InitMap> LocalInitMap;
typedef PerThreadStorage<QpMapFreq> LocalQpMapFreq; // PerThreadStorage: thread-local quick pattern map
typedef PerThreadStorage<CgMapFreq> LocalCgMapFreq; // PerThreadStorage: thread-local canonical pattern map
typedef PerThreadStorage<QpMapDomain> LocalQpMapDomain;
typedef PerThreadStorage<CgMapDomain> LocalCgMapDomain;

class EdgeMiner : public Miner {
public:
	EdgeMiner(Graph *g, unsigned size = 3, int nthreads = 1) {
		graph = g;
		max_size = size;
		numThreads = nthreads;
		construct_edgemap();
		init_localmaps.set_size(nthreads);
		qp_localmaps.set_size(nthreads);
		cg_localmaps.set_size(nthreads);
	}
	virtual ~EdgeMiner() {}
	void extend_edge(unsigned level, EmbeddingList& emb_list) {
		UintList num_new_emb(emb_list.size());
		#pragma omp parallel for
		for (size_t pos = 0; pos < emb_list.size(); pos ++) {
			EdgeEmbedding emb(level+1);
			get_embedding(level, pos, emb_list, emb);
			num_new_emb[pos] = 0;
			unsigned n = emb.size();
			std::set<vidType> vert_set;
			if (n > 3)
				for (unsigned i = 0; i < n; i ++) vert_set.insert(emb.get_vertex(i));
			for (unsigned i = 0; i < n; ++i) {
				vidType src = emb.get_vertex(i);
				if (emb.get_key(i) == 0) { // TODO: need to fix this
					IndexT row_begin = graph->edge_begin(src); 
					IndexT row_end = graph->edge_end(src); 
					for (IndexT e = row_begin; e < row_end; e++) {
						IndexT dst = graph->getEdgeDst(e);
						BYTE existed = 0;
						//if (is_frequent_edge[e])
							if (!is_edge_automorphism(n, emb, i, src, dst, existed, vert_set))
								num_new_emb[pos] ++;
					}
				}
			}
			emb.clean();
		}
		Ulong new_size = std::accumulate(num_new_emb.begin(), num_new_emb.end(), (Ulong)0);
		std::cout << "new_size = " << new_size << "\n";
		assert(new_size < 4294967296); // TODO: currently do not support vector size larger than 2^32
		UintList indices = parallel_prefix_sum(num_new_emb);
		new_size = indices[indices.size()-1];
		emb_list.add_level(new_size);
		#pragma omp parallel for
		for (size_t pos = 0; pos < emb_list.size(level); pos ++) {
			EdgeEmbedding emb(level+1);
			get_embedding(level, pos, emb_list, emb);
			unsigned start = indices[pos];
			unsigned n = emb.size();
			std::set<vidType> vert_set;
			if (n > 3)
				for (unsigned i = 0; i < n; i ++) vert_set.insert(emb.get_vertex(i));
			for (unsigned i = 0; i < n; ++i) {
				IndexT src = emb.get_vertex(i);
				if (emb.get_key(i) == 0) {
					IndexT row_begin = graph->edge_begin(src); 
					IndexT row_end = graph->edge_end(src); 
					for (IndexT e = row_begin; e < row_end; e++) {
						IndexT dst = graph->getEdgeDst(e);
						BYTE existed = 0;
						//if (is_frequent_edge[e])
							if (!is_edge_automorphism(n, emb, i, src, dst, existed, vert_set)) {
								emb_list.set_idx(level+1, start, pos);
								emb_list.set_his(level+1, start, i);
								emb_list.set_vid(level+1, start++, dst);
							}
					}
				}
			}
		}
	}
	inline unsigned init_aggregator() {
		init_map.clear();
		for (IndexT src = 0; src < graph->num_vertices(); src ++) {
			InitMap *lmap = init_localmaps.getLocal();
			auto src_label = graph->getData(src);
			IndexT row_begin = graph->edge_begin(src);
			IndexT row_end = graph->edge_end(src);
			for (IndexT e = row_begin; e < row_end; e++) {
				IndexT dst = graph->getEdgeDst(e);
				auto dst_label = graph->getData(dst);
				if (src_label <= dst_label) {
					InitPattern key = get_init_pattern(src_label, dst_label);
					if (lmap->find(key) == lmap->end()) {
						(*lmap)[key] = new DomainSupport(2);
						(*lmap)[key]->set_threshold(threshold);
					}
					(*lmap)[key]->add_vertex(0, src);
					(*lmap)[key]->add_vertex(1, dst);
				}
			}
		}
		merge_init_map();
		std::cout << "Number of single-edge patterns: " << init_map.size() << "\n";
		unsigned count = 0;
		for (auto it = init_map.begin(); it != init_map.end(); ++it)
			if (it->second->get_support()) count ++;
		return count; // return number of frequent single-edge patterns
	}
	inline void quick_aggregate(unsigned level, EmbeddingList& emb_list) {
		for (auto i = 0; i < numThreads; i++) qp_localmaps.getLocal(i)->clear();
		#pragma omp parallel for
		for (size_t pos = 0; pos < emb_list.size(); pos ++) {
			QpMapDomain *lmap = qp_localmaps.getLocal();
			EdgeEmbedding emb(level+1);
			get_embedding(level, pos, emb_list, emb);
			unsigned n = emb.size();
			QPattern qp(emb, true);
			bool qp_existed = false;
			auto it = lmap->find(qp);
			if (it == lmap->end()) {
				(*lmap)[qp] = new DomainSupport(n);
				(*lmap)[qp]->set_threshold(threshold);
				emb_list.set_pid(pos, qp.get_id());
			} else {
				qp_existed = true;
				emb_list.set_pid(pos, (it->first).get_id());
			}
			for (unsigned i = 0; i < n; i ++) {
				if ((*lmap)[qp]->has_domain_reached_support(i) == false)
					(*lmap)[qp]->add_vertex(i, emb.get_vertex(i));
			}
			if (qp_existed) qp.clean();
		}
	}
	void insert_id_map(int qp_id, int cg_id) {
		std::unique_lock<std::mutex> lock(map_mutex);
		id_map.insert(std::make_pair(qp_id, cg_id));
	}
	// aggregate quick patterns into canonical patterns.
	// construct id_map from quick pattern ID (qp_id) to canonical pattern ID (cg_id)
	void canonical_aggregate() {
		id_map.clear();
		for (auto i = 0; i < numThreads; i++) cg_localmaps.getLocal(i)->clear();
		for (std::pair<QPattern, DomainSupport*> element : qp_map) {
			CgMapDomain *lmap = cg_localmaps.getLocal();
			unsigned num_domains = element.first.get_size();
			CPattern cg(element.first);
			int qp_id = element.first.get_id();
			int cg_id = cg.get_id();
			insert_id_map(qp_id, cg_id);
			auto it = lmap->find(cg);
			if (it == lmap->end()) {
				(*lmap)[cg] = new DomainSupport(num_domains);
				(*lmap)[cg]->set_threshold(threshold);
				element.first.set_cgid(cg.get_id());
			} else {
				element.first.set_cgid((it->first).get_id());
			}
			VertexPositionEquivalences equivalences;
			element.first.get_equivalences(equivalences);
			for (unsigned i = 0; i < num_domains; i ++) {
				if ((*lmap)[cg]->has_domain_reached_support(i) == false) {
					unsigned qp_idx = cg.get_quick_pattern_index(i);
					assert(qp_idx >= 0 && qp_idx < num_domains);
					UintSet equ_set = equivalences.get_equivalent_set(qp_idx);
					for (unsigned idx : equ_set) {
						DomainSupport *support = element.second;
						if (support->has_domain_reached_support(idx) == false) {
							bool reached_threshold = (*lmap)[cg]->add_vertices(i, support->domain_sets[idx]);
							if (reached_threshold) break;
						} else {
							(*lmap)[cg]->set_domain_frequent(i);
							break;
						}
					}
				}
			}
			cg.clean();
		}
	}
	inline void merge_init_map() {
		init_map = *(init_localmaps.getLocal(0));
		for (auto i = 1; i < numThreads; i++) {
			for (auto element : *init_localmaps.getLocal(i)) {
				DomainSupport *support = element.second;
				if (init_map.find(element.first) == init_map.end()) {
					init_map[element.first] = support;
				} else {
					for (unsigned i = 0; i < 2; i ++) {
						if (!init_map[element.first]->has_domain_reached_support(i)) {
							if (support->has_domain_reached_support(i))
								init_map[element.first]->set_domain_frequent(i);
							else init_map[element.first]->add_vertices(i, support->domain_sets[i]);
						}
					}
				}
			}
		}
	}
	inline void merge_qp_map(unsigned num_domains) {
		qp_map.clear();
		qp_map = *(qp_localmaps.getLocal(0));
		for (auto i = 1; i < numThreads; i++) {
			const QpMapDomain *lmap = qp_localmaps.getLocal(i);
			for (auto element : *lmap) {
				if (qp_map.find(element.first) == qp_map.end())
					qp_map[element.first] = element.second;
			}
			for (std::pair<QPattern, DomainSupport*> element : *lmap) {
				DomainSupport *support = element.second;
				for (unsigned i = 0; i < num_domains; i ++) {
					if (!qp_map[element.first]->has_domain_reached_support(i) && qp_map[element.first] != support) {
						if (support->has_domain_reached_support(i))
							qp_map[element.first]->set_domain_frequent(i);
						else qp_map[element.first]->add_vertices(i, support->domain_sets[i]);
					}
				}
			}
		}
	}
	inline void merge_cg_map(unsigned num_domains) {
		cg_map.clear();
		cg_map = *(cg_localmaps.getLocal(0));
		for (auto i = 1; i < numThreads; i++) {
			const CgMapDomain *lmap = cg_localmaps.getLocal(i);
			for (auto element : *lmap) {
				if (cg_map.find(element.first) == cg_map.end())
					cg_map[element.first] = element.second;
			}
			for (std::pair<CPattern, DomainSupport*> element : *lmap) {
				DomainSupport *support = element.second;
				for (unsigned i = 0; i < num_domains; i ++) {
					if (!cg_map[element.first]->has_domain_reached_support(i) && cg_map[element.first] != support) {
						if (support->has_domain_reached_support(i))
							cg_map[element.first]->set_domain_frequent(i);
						else cg_map[element.first]->add_vertices(i, support->domain_sets[i]);
					}
				}
			}
		}
	}

	// Filtering for FSM
#ifdef ENABLE_LABEL
	inline void init_filter(EmbeddingList& emb_list) {
		UintList is_frequent_emb(emb_list.size(), 0);
		#pragma omp parallel for
		for (size_t pos = 0; pos < emb_list.size(); pos ++) {
			vidType src = emb_list.get_idx(1, pos);
			vidType dst = emb_list.get_vid(1, pos);
			auto src_label = graph->getData(src);
			auto dst_label = graph->getData(dst);
			InitPattern key = get_init_pattern(src_label, dst_label);
			if (init_map[key]->get_support()) is_frequent_emb[pos] = 1;
		}

		//assert(emb_list.size()*2 == graph->num_edges()); // symmetric graph
		is_frequent_edge.resize(graph->num_edges());
		std::fill(is_frequent_edge.begin(), is_frequent_edge.end(), 0);
		#pragma omp parallel for
		for (size_t pos = 0; pos < emb_list.size(); pos ++) {
			if (is_frequent_emb[pos]) {
				vidType src = emb_list.get_idx(1, pos);
				vidType dst = emb_list.get_vid(1, pos);
				unsigned eid0 = edge_map[OrderedEdge(src,dst)];
				unsigned eid1 = edge_map[OrderedEdge(dst,src)];
				__sync_bool_compare_and_swap(&is_frequent_edge[eid0], 0, 1);
				__sync_bool_compare_and_swap(&is_frequent_edge[eid1], 0, 1);
			}
		}
		std::cout << "Number of frequent edges: " << count(is_frequent_edge.begin(), is_frequent_edge.end(), 1) << "\n";
	
		UintList indices = parallel_prefix_sum(is_frequent_emb);
		auto vid_list0 = emb_list.get_idx_list(1);
		auto vid_list1 = emb_list.get_vid_list(1);
		#pragma omp parallel for
		for (size_t pos = 0; pos < emb_list.size(); pos ++) {
			if (is_frequent_emb[pos]) {
				vidType src = vid_list0[pos];
				vidType dst = vid_list1[pos];
				unsigned start = indices[pos];
				emb_list.set_vid(1, start, dst);
				emb_list.set_idx(1, start, src);
			}
		}
		emb_list.remove_tail(indices.back());
	}
#endif
	inline void filter(unsigned level, EmbeddingList &emb_list) {
		UintList is_frequent_emb(emb_list.size(), 0);
		#pragma omp parallel for
		for (size_t pos = 0; pos < emb_list.size(); pos ++) {
			unsigned qp_id = emb_list.get_pid(pos);
			unsigned cg_id = id_map.at(qp_id);
			if (domain_support_map.at(cg_id))
				is_frequent_emb[pos] = 1;
		}
		UintList indices = parallel_prefix_sum(is_frequent_emb);
		VertexList vid_list = emb_list.get_vid_list(level);
		UintList idx_list = emb_list.get_idx_list(level);
		ByteList his_list = emb_list.get_his_list(level);
		for (size_t pos = 0; pos < emb_list.size(); pos ++) {
			if (is_frequent_emb[pos]) {
				unsigned start = indices[pos];
				vidType vid = vid_list[pos];
				IndexTy idx = idx_list[pos];
				BYTE his = his_list[pos];
				emb_list.set_idx(level, start, idx);
				emb_list.set_vid(level, start, vid);
				emb_list.set_his(level, start, his);
			}
		}
		emb_list.remove_tail(indices.back());
	}
	inline void set_threshold(const unsigned minsup) { threshold = minsup; }
	inline void printout_agg(const CgMapFreq &cg_map) {
		for (auto it = cg_map.begin(); it != cg_map.end(); ++it)
			std::cout << "{" << it->first << " --> " << it->second << std::endl;
	}
	inline void printout_agg() {
		std::cout << "num_patterns: " << cg_map.size() << " num_quick_patterns: " << qp_map.size() << "\n";
		BoolVec support(cg_map.size());
		int i = 0;
		for (auto it = cg_map.begin(); it != cg_map.end(); ++it) {
			support[i] = it->second->get_support();
			i ++;
		}
		i = 0;
		for (auto it = cg_map.begin(); it != cg_map.end(); ++it) {
			std::cout << "{" << it->first << " --> " << support[i] << std::endl;
			i ++;
		}
	}
	inline unsigned support_count() {
		domain_support_map.clear();
		unsigned count = 0;
		for (auto it = cg_map.begin(); it != cg_map.end(); ++it) {
			bool support = it->second->get_support();
			domain_support_map.insert(std::make_pair(it->first.get_id(), support));
			if (support) count ++;
		}
		return count;
	}
	// construct edge-map for later use. May not be necessary if Galois has this support
	void construct_edgemap() {
		for (auto src = 0; src < graph->num_vertices(); src ++) {
			IndexT row_begin = graph->edge_begin(src);
			IndexT row_end = graph->edge_end(src);
			for (IndexT e = row_begin; e < row_end; e++) {
				auto dst = graph->getEdgeDst(e);
				OrderedEdge edge(src, dst);
				edge_map.insert(std::pair<OrderedEdge, unsigned>(edge, e));
			}
		}
	}

private:
	unsigned threshold;
	InitMap init_map;
	UintMap id_map;
	unsigned max_size;
	int numThreads;
	FreqMap freq_support_map;
	DomainMap domain_support_map;
	std::map<OrderedEdge, unsigned> edge_map;
	std::set<std::pair<vidType,vidType> > freq_edge_set;
	std::vector<unsigned> is_frequent_edge;
	LocalInitMap init_localmaps; // initialization map, only used for once, no need to clear
	LocalQpMapDomain qp_localmaps; // quick pattern local map for each thread
	LocalCgMapDomain cg_localmaps; // canonical pattern local map for each thread
	QpMapDomain qp_map; // quick pattern map
	CgMapDomain cg_map; // canonical graph map
	std::mutex map_mutex;

	inline InitPattern get_init_pattern(BYTE src_label, BYTE dst_label) {
		if (src_label <= dst_label) return std::make_pair(src_label, dst_label);
		else return std::make_pair(dst_label, src_label);
	}
	inline void get_embedding(unsigned level, unsigned pos, const EmbeddingList& emb_list, EdgeEmbedding &emb) {
		vidType vid = emb_list.get_vid(level, pos);
		IndexTy idx = emb_list.get_idx(level, pos);
		BYTE his = emb_list.get_his(level, pos);
		BYTE lab = graph->getData(vid);
		ElementType ele(vid, 0, lab, his);
		emb.set_element(level, ele);
		for (unsigned l = 1; l < level; l ++) {
			vid = emb_list.get_vid(level-l, idx);
			his = emb_list.get_his(level-l, idx);
			lab = graph->getData(vid);
			ElementType ele(vid, 0, lab, his);
			emb.set_element(level-l, ele);
			idx = emb_list.get_idx(level-l, idx);
		}
		lab = graph->getData(idx);
		ElementType ele0(idx, 0, lab, 0);
		emb.set_element(0, ele0);
	}
	bool is_quick_automorphism(unsigned size, const EdgeEmbedding& emb, BYTE history, vidType src, vidType dst, BYTE& existed) {
		if (dst <= emb.get_vertex(0)) return true;
		if (dst == emb.get_vertex(1)) return true;
		if (history == 0 && dst < emb.get_vertex(1)) return true;
		if (size == 2) {
		} else if (size == 3) {
			if (history == 0 && emb.get_history(2) == 0 && dst <= emb.get_vertex(2)) return true;
			if (history == 0 && emb.get_history(2) == 1 && dst == emb.get_vertex(2)) return true;
			if (history == 1 && emb.get_history(2) == 1 && dst <= emb.get_vertex(2)) return true;
			if (dst == emb.get_vertex(2)) existed = 1;
			//if (!existed && max_size < 4) return true;
		} else {
			std::cout << "Error: should go to detailed check\n";
		}
		return false;
	}
	bool is_edge_automorphism(unsigned size, const EdgeEmbedding& emb, BYTE history, vidType src, vidType dst, BYTE& existed, const std::set<vidType>& vertex_set) {
		if (size < 3) return is_quick_automorphism(size, emb, history, src, dst, existed);
		// check with the first element
		if (dst <= emb.get_vertex(0)) return true;
		if (history == 0 && dst <= emb.get_vertex(1)) return true;
		// check loop edge
		if (dst == emb.get_vertex(emb.get_history(history))) return true;
		if (vertex_set.find(dst) != vertex_set.end()) existed = 1;
		// check to see if there already exists the vertex added; 
		// if so, just allow to add edge which is (smaller id -> bigger id)
		if (existed && src > dst) return true;
		std::pair<vidType, vidType> added_edge(src, dst);
		for (unsigned index = history + 1; index < emb.size(); ++index) {
			std::pair<vidType, vidType> edge;
			edge.first = emb.get_vertex(emb.get_history(index));
			edge.second = emb.get_vertex(index);
			//assert(edge.first != edge.second);
			int cmp = compare(added_edge, edge);
			if(cmp <= 0) return true;
		}
		return false;
	}
	inline void swap(std::pair<vidType, vidType>& pair) {
		if (pair.first > pair.second) {
			vidType tmp = pair.first;
			pair.first = pair.second;
			pair.second = tmp;
		}
	}
	inline int compare(std::pair<vidType, vidType>& oneEdge, std::pair<vidType, vidType>& otherEdge) {
		swap(oneEdge);
		swap(otherEdge);
		if(oneEdge.first == otherEdge.first) return oneEdge.second - otherEdge.second;
		else return oneEdge.first - otherEdge.first;
	}
};

#endif // EDGE_MINER_HPP_
