#pragma GCC optimize("O3,unroll-loops")
#include "graph.h"
#define ll uint64_t 
#include <cblas.h>

// adjacency arrays high vertices
const ll mx_sz = 10000;
float adj_arr_h[mx_sz*mx_sz];
float adj_arr_h_2[mx_sz*mx_sz];

const ll mx_label_sz = 10000000; // max label size
vidType h[mx_label_sz];
vidType label_to_h[mx_label_sz];
vidType type[mx_label_sz];

// naive matmul
void MM(vector<vector<vidType>> &A, vector<vector<vidType>> &B){
	vidType n = A.size();
	assert(n == (vidType)B.size());

	vector<vector<vidType>> C(n, vector<vidType>(n));
	
	for (vidType i = 0; i < n; i++){
		for (vidType j = 0; j < n; j++){
			vidType tot = 0;	
			#pragma omp parallel for reduction(+ : tot) schedule(dynamic, 1)			
			for (vidType k = 0; k < n; k++){
				tot += A[i][k]*B[k][j];
			}
			C[i][j] = tot;
		}
	}
	A=C;
}

// count intersections when adjacency graph is sorted by degree
ll intersect(vidType &x, vidType &y, vector<vidType> &A, vector<vidType> &B, vector<vidType> &type){
    ll cnt = 0;
    vidType Asz = (vidType)A.size();
    vidType Bsz = (vidType)B.size();
    vidType idx1 = 0;
    vidType idx2 = 0;
    while(idx1 < Asz && idx2 < Bsz){
      if (A[idx1] > B[idx2]) idx2++;
      else if (A[idx1] < B[idx2]) idx1++;
      else cnt++;
    }
    return cnt;
}


// count intersections when adjacency graph is not sorted by degree 
// HHL are counted 2x
// HLL are counted 4x
// LLL are counted 6x
// so we weight the counter differently according to the type
ll intersect_unsorted(vidType &x, vidType &y, vector<vidType> &A, vector<vidType> &B, vector<vidType> &type){
	// size of A < size of B
	ll cnt=0;
	vidType Asz = (vidType)A.size();
	vidType Bsz = (vidType)B.size();
	vidType idx1 = 0;
	vidType idx2 = 0;
	vidType h_cnt = type[x]+type[y];
	while(idx1 < Asz && idx2 < Bsz){
		if (A[idx1] > B[idx2]) idx2++;
		else if (A[idx1] < B[idx2]) idx1++;
		else {
			vidType h_tot = h_cnt+type[A[idx1]];
			if (h_tot == 0)	cnt += 2;
			else if (h_tot == 1) cnt += 3;
			else cnt += 6;
			idx1++;
			idx2++;
		}
	}
	return cnt;
}


// MM HELPER FUNCTIONS ------------
//! wrapper function to call cblas_sgemm
void sgemm_cpu(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
               const int M, const int N, const int K, const float alpha,
               const float* A, const float* B, const float beta, float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
	cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

// A: x*z; B: z*y; C: x*y
// C = A*B: 
// matmul(n, n, n, A, B, C, 0, 0, 0);
// ROWMAJOR: A[0]=A[0][0], A[1] = A[0][1], A[2] = A[1][0], A[3] = A[1][1]
// input matrices are flattened
void matmul(const size_t dim_x, const size_t dim_y, const size_t dim_z,
            const float* A, const float* B, float* C, bool transA, bool transB, bool accum) {
  const CBLAS_TRANSPOSE TransA = transA ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE TransB = transB ? CblasTrans : CblasNoTrans;
  sgemm_cpu(TransA, TransB, dim_x, dim_y, dim_z, 1.0, A, B, accum?1.0:0.0, C);
}
// --------------------------------
// --------------------------------

void TCSolver(Graph &g, uint64_t &total, int, int, int threshold) {
	int num_threads = 1;
	int openblas_num_threads = openblas_get_num_threads();  
	#pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP TC (" << num_threads << " threads)\n";
  cout << "with Openblas (" << openblas_num_threads << " threads)\n";
	
	Timer t;
	t.Start();
	vidType n = g.V();
	
	Timer p;
	p.Start();
	// build adjacency lists
	vector<vector<vidType>> adj(n);
	
  // need to sort adjacency lists by size. otherwise, triangles of the form 
  // H --> L--> L (H --------> L on the bottom) 
  // are not counted correctly (they should not exist)
  // assuming that g.N(i) was created such that lower vertices (in the non-DAG form) always point towards higher vertices, 
  // we need to determine H and L vertices based on their degrees in the non-DAG form
  // so, we cannot avoid this pre-processing unless we pass in degree information from the non-DAG form (which should be possible)
  for (vidType i = 0; i < n; i++){
		auto ni = g.N(i);
	  for (auto u : ni){
			adj[i].push_back(u);
      adj[u].push_back(i);
	  }
	}
	// collect vertices of high degree
  // these degrees are all with respect to the non-DAG version of the graph
	// double alpha = 2.81; // coefficient of matrix multiplication
	// vidType D = pow(g.E(),(alpha-1.0)/(alpha+1.0)); // theoretically optimal threshold, usually not practically applicable
	vidType D = threshold;
	cout << "threshold degree: " << D << endl;
  vidType cur = 0;
	for (vidType i = 0; i < n; i++){
		if ((vidType)adj[i].size() >= D){
      h[cur]=i;
      label_to_h[i]=cur;
      type[i]=1;
      cur++;
		}
	}
	vidType m = cur;
	cout << "number of vertices with high degree: " << m << endl;

	p.Stop();
	cout << "preprocessing runtime = " << p.Seconds() << " sec\n";
	Timer s;
	s.Start();
   
  Timer low_counter_timer;
  low_counter_timer.Start();
	// triangles with at least one low degree
	ll lowcounter = 0;
  #pragma omp parallel for reduction(+ : lowcounter) schedule(dynamic, 1)
  for (vidType i = 0; i < n; i++) {
  //  if (!type[i]){
      auto ni = g.N(i); 
      for (auto v : ni) {
        lowcounter += (!type[i])*(uint64_t)intersection_num(ni, g.N(v));
      }    
   // }
  }
  low_counter_timer.Stop();
  Timer high_counter_timer;
  high_counter_timer.Start();

	// triangles with all high degrees
	ll highcounter = 0;
	
  for (vidType i = 0; i < m; i++){
    for (vidType j : g.N(h[i])){
      if (type[j]==1){
        adj_arr_h[m*i+label_to_h[j]]=1;
      }
    }
  }

  matmul(m, m, m, adj_arr_h, adj_arr_h, adj_arr_h_2, 0, 0, 0);
  
  #pragma omp parallel for reduction(+ : highcounter) schedule(dynamic, 1)
	for (vidType i = 0; i < m; i++){
		for (vidType j = 0; j < m; j++){
			highcounter += adj_arr_h_2[i*m+j] * adj_arr_h[i*m+j];
		}
	}
  
  high_counter_timer.Stop();
	// compute final answer
	cout << "lowcnt: " << lowcounter << endl;
	cout << "highcnt: " << highcounter << endl;
	total = lowcounter + highcounter;
	s.Stop();
	t.Stop();
  cout << "time to compute low triangles = " << low_counter_timer.Seconds() << " sec\n";
  cout << "time to compute high triangles = " << high_counter_timer.Seconds() << " sec\n";
	cout << "computing runtime = " << s.Seconds() << " sec\n";
	std::cout << "total runtime [omp_a0] = " << t.Seconds() << " sec\n";
	
  // the "computing runtime" assumes the following quantities are already given: 
  // -compressed list of high degree nodes, where high degree nodes are determined by their degree in the non-DAG version of the graph
  // adjacency arrays in DAG version of the graph
  return;
}


