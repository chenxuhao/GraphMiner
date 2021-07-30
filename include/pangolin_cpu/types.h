#ifndef TYPES_H
#define TYPES_H
#include <omp.h>

template <typename T>
class PerThreadStorage {
protected:
	unsigned offset;
	unsigned num_threads;
	T **p_data;
public:
	PerThreadStorage() {}
	~PerThreadStorage() {}
	T* getLocal() {
		int tid = omp_get_thread_num();
		return p_data[tid];
	}
	T* getLocal(unsigned tid) {
		return p_data[tid];
	}
	size_t size() const {  return num_threads; }
	void set_size(unsigned nthreads) {
		num_threads = nthreads;
		p_data = new T*[nthreads];
		for (unsigned i = 0; i < num_threads; i++)
			p_data[i] = new T[1];
	}
	void init(T value) {
		for (unsigned i = 0; i < num_threads; i++)
			*(p_data[i]) = value;
	}
};

template <typename T>
class Accumulator {
public:
	Accumulator() {}
	Accumulator(int nthreads) {
		m_data.set_size(nthreads);
		m_data.init(0);
	}
	~Accumulator() {}
	void resize(int nthreads) {
		m_data.set_size(nthreads);
		m_data.init(0);
	}
	Accumulator& operator+=(const T& rhs) {
		T& lhs = *m_data.getLocal();
		lhs += rhs;
		return *this;
	}
	T reduce() {
		T d0 = 0;
		for (size_t i = 0; i < m_data.size(); i++)
			d0 += *m_data.getLocal(i);
		return d0;
	}
protected:
	PerThreadStorage<T> m_data;
	//int num_threads;
};

// We provide two types of 'support': frequency and domain support.
// Frequency is used for counting, e.g. motif counting.
// Domain support, a.k.a, the minimum image-based support, is used for FSM. It has the anti-monotonic property.
typedef float MatType;
typedef unsigned Frequency;
typedef std::vector<std::vector<MatType> > Matrix;
typedef Accumulator<unsigned> UintAccu;
typedef Accumulator<unsigned long> UlongAccu;
typedef std::unordered_map<unsigned, unsigned> UintMap;
typedef PerThreadStorage<UintMap> LocalUintMap;

#endif
