#include "graph.h"
#define ll uint64_t
#include <cblas.h>

// adjacency arrays
const ll mx_sz = 10000; // change to fit graph
float adj_arr_h[mx_sz*mx_sz]; // adjacency array for high vertices
float adj_arr_h_2[mx_sz*mx_sz]; // squared adjacency array for high vertices
float adj_x[mx_sz*mx_sz]; // adjacency array for neighbors of node x
float adj_x_2[mx_sz*mx_sz]; // squared adjacency array for neighbors of node x

const ll mx_label_sz = 10000000; // max label size
vidType nx[mx_label_sz]; // neighbors of each x
vidType label_to_nx[mx_label_sz]; // original labels to index in neighbors array
vidType h[mx_label_sz]; // high vertices
vidType label_to_h[mx_label_sz]; // original labels to index in high vertices array
vidType is_neighbor[mx_label_sz];

// naive matrix multiplication
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

void cnt_3(Graph &g, vector<vector<vidType>> &adj, vector<vidType> &type, vidType &n, vidType &m, uint64_t &total_3){
	// counts all HH diameters with at least one L other
	// O((e/D)^2*e)?
	cout << "running count 3..." << endl;
	//vector<vector<vidType>> adj_arr_h(m, vector<vidType>(m));
  //for (vidType i = 0; i < m; i++){
   // for (vidType j : adj[h[i]]){
   //   if (type[j]==1){
   //     adj_arr_h[i][label_to_h[j]] = 1;
   //     adj_arr_h[label_to_h[j]][i] = 1;
  //    }
  //  }
  //}

  // #pragma omp parallel for reduction(+ : total_3) schedule(dynamic, 1)
  for (vidType i = 0; i < m; i++){
		// #pragma omp parallel for reduction(+ : total_3) schedule(dynamic, 1)
    for (vidType j = 0; j < m; j++){
			if (adj_arr_h[i*m+j]==1){
        vidType idx_1 = 0, idx_2 = 0;
        uint64_t h_cnt=0, cnt=0;
        while(idx_1 < (vidType)adj[h[i]].size() && idx_2 < (vidType)adj[h[j]].size()){
          if (adj[h[i]][idx_1] == adj[h[j]][idx_2]){
            h_cnt+=type[adj[h[i]][idx_1]];
            cnt+=1;
            idx_1++;
            idx_2++;
          } else if (adj[h[i]][idx_1] < adj[h[j]][idx_2]) idx_1++;
          else idx_2++;
        }
        if (cnt > 1){
          total_3 += cnt*(cnt-1)/2;
          if (h_cnt > 1) total_3 -= h_cnt*(h_cnt-1)/2;
        } 
      }
		}
	}
	total_3 /= 2;
}

void cnt_2(Graph &g, vector<vector<vidType>> &adj, vector<vidType> &type, vidType &n, vidType &useless, uint64_t &total_2){
	// counts all HL and LL diameters with any others
	// O(sum deg_L^alpha)\in D^(alpha-1)*e, summing over adjacency graphs of all low degree vertices
	cout << "running count 2..." << endl;
  
  vidType neighbor_index = 100;
  // #pragma omp parallel for reduction(+ : total_2) schedule(dynamic, 100)
	for (vidType i = 0; i < n; i++){
		// cout << i << endl;
    neighbor_index++;
    if (type[i]==0){
			// cout << "found a low degree vertex, with degree: " << (vidType)adj[i].size() << '\n';
			// vector<vidType> nx;
		  // vector<vidType> label_to_nx(n);
			// nx.push_back(i);
			//label_to_nx[i] = 0;
			//for (auto u : adj[i]){
			//	nx.push_back(u);
		  //	label_to_nx[u] = nx.size()-1;
			//  is_neighbor[u]=i;
      //}
      
      is_neighbor[i]=neighbor_index;
      nx[0]=i;
      label_to_nx[i]=0;
      //cout << "this is another flag" << endl;
      vidType cur = 1;
      for (auto u : adj[i]){
        nx[cur] = u;
        label_to_nx[u] = cur;
        is_neighbor[u]=neighbor_index;
        cur++;
     }

			// vidType m = nx.size();
     vidType m = cur;
      for (vidType k = 0; k <= m*m; k++){
        adj_x[k]=0;
        adj_x_2[k]=0;
      }
//      vector<vector<vidType>> adj_x(m, vector<vidType>(m));
//      for (vidType k = 0; k < m; k++){
//        for (vidType j : adj[nx[k]]){
//          if (is_neighbor[j] != i) continue;
 //         adj_x[k][label_to_nx[j]]=1;
   //       adj_x[label_to_nx[j]][k]=1;
 //       }  
   //   }
      for (vidType k = 0; k < m; k++){
		  	for (vidType j : adj[nx[k]]){
          if (is_neighbor[j] != neighbor_index) continue;
					adj_x[k*m+label_to_nx[j]] = 1;
          adj_x[k+label_to_nx[j]*m] = 1;
				}
			}
			//vector<vector<vidType>> adj_x_2=adj_x;
			//MM(adj_x_2, adj_x);
			
      matmul(m, m, m, adj_x, adj_x, adj_x_2, 0,0,0);
      for (auto u : adj[i]){
				if (type[u] == 1){ // HL
					total_2 += adj_x_2[label_to_nx[i]+label_to_nx[u]*m]*(adj_x_2[label_to_nx[i]+label_to_nx[u]*m]-1);
				} else { // LL
					total_2 += (adj_x_2[label_to_nx[i]+label_to_nx[u]*m]*(adj_x_2[label_to_nx[i]+label_to_nx[u]*m]-1))/2;
				}
			}
      //for (auto u : adj[i]){
       // if (type[u] == 1){ // HL
      //    total_2 += adj_x_2[label_to_nx[i]][label_to_nx[u]]*(adj_x_2[label_to_nx[i]][label_to_nx[u]]-1);
      //  } else { // LL
      //    total_2 += adj_x_2[label_to_nx[i]][label_to_nx[u]]*(adj_x_2[label_to_nx[i]][label_to_nx[u]]-1)/2;
      //  }
     // }
			// cout << "new total 2 value: " << total_2 << '\n';
		}
	}
	total_2/=2;	
}

void cnt_1(Graph &g, vector<vector<vidType>> &adj, vector<vidType> &type, vidType &n, vidType &m, uint64_t &total_1){
  // counts all HH diameters with HH others
	// O(sum deg_H^alpha)\in O(deg_H*sum deg_H^(alpha-1))\in O(e^alpha/D^(alpha-1))	
	cout << "running count 1..." << endl;
	
	// vector<vector<vidType>> adj_arr_h(m, vector<vidType>(m));
	//for (vidType i = 0; i < m; i++){
	//	for (vidType j : adj[h[i]]){
	//		if (type[j]==1){
	//			adj_arr_h[i*m+label_to_h[j]] = 1;
	//			adj_arr_h[label_to_h[j]*m+i] = 1;
	//		}
	//	}
	//}
	
	// vector<vector<vidType>> adj_arr_h_2=adj_arr_h;
	// MM(adj_arr_h_2, adj_arr_h);

  matmul(m, m, m, adj_arr_h, adj_arr_h, adj_arr_h_2, 0, 0, 0);
  
//  #pragma omp parallel for reduction(+ : total_1) schedule(dynamic, 100000)
	for (vidType i = 0; i < m; i++){
//		#pragma omp parallel for reduction(+ : total_1) schedule(dynamic, 1)
    for (vidType j = 0; j < m; j++){
			if (adj_arr_h_2[i*m+j] && adj_arr_h[i*m+j]){
				total_1 += adj_arr_h_2[i*m+j]*(adj_arr_h_2[i*m+j]-1)/2;
			}	
		}
	}
	// total_1/=2;
}

void diamondSolver(Graph &g, int k, uint64_t &total){
	// assert(k==4);
	Timer construction;
  construction.Start();
  vidType n = g.V();
	vector<vector<vidType>> adj(n);
  for (vidType i = 0; i < n; i++){
    auto ni = g.N(i);
    for (auto u : ni){
      adj[i].push_back(u);
    }
  }
  construction.Stop();
  cout << "construction time: " << construction.Seconds() << '\n';

	vidType D = pow(g.E(), 0.5); // TODO: what is the best possible threshold? 
	//D=150; // for testing
	cout << "number of edges: " << g.E() << '\n';
	cout << "threshold for high degree vertices: " << D << '\n';
	vidType num_high_deg = 0;
	vector<vidType> type(n);
	for (vidType i = 0; i < n; i++){
		if ((vidType)adj[i].size() >= D) {type[i] = 1; num_high_deg++;}
	}
	cout << "number of high degree vertices: " << num_high_deg << '\n';

  // compute graphs for high vertices
  vidType cur = 0;
  for (vidType i = 0; i < n; i++){
    if (type[i]==1){
      h[cur]=i;
      label_to_h[i] = cur;
      cur++;
    }
  }
  vidType m = cur;
  ll non_zero = 0;
  for (vidType i = 0; i < m; i++){
    for (vidType j : adj[h[i]]){
      if (type[j]==1){
        non_zero++;
        adj_arr_h[i*m+label_to_h[j]] = 1;
        adj_arr_h[label_to_h[j]*m+i] = 1;
      }
    }
  }
  cout << "NON ZERO: " << non_zero << " TOTAL SIZE: " << m*m << '\n';
	// totals
	uint64_t total_1 = 0, total_2 = 0, total_3 = 0;
	Timer time_cnt_1;
  time_cnt_1.Start();
  cnt_1(g, adj, type, n, m, total_1);
  time_cnt_1.Stop();
  total_1 /= 2;
  Timer time_cnt_2;
  time_cnt_2.Start();
	cnt_2(g, adj, type, n, m, total_2);
  time_cnt_2.Stop();
  Timer time_cnt_3;
  time_cnt_3.Start();
	cnt_3(g, adj, type, n, m, total_3);
  time_cnt_3.Stop();
	total = total_1 + total_2 + total_3; 
	cout << "HHHH total: " << total_1 << '\n';
	cout << "HL/LL diameter total: " << total_2 << '\n';
	cout << "HH diamater with>=1 L total: " << total_3 << '\n';
  cout << "time to compute HHHH: " << time_cnt_1.Seconds() << '\n';
  cout << "time to compute HL/LL: " << time_cnt_2.Seconds() << '\n';
  cout << "time to compute HH diameters with >= L: " << time_cnt_3.Seconds() << '\n';
}

void DiamondSolver(Graph &g, int k, uint64_t &total, int, int){
	int num_threads = 1;
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	std::cout << "OpenMP diamond listing (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  double start_time = omp_get_wtime();
	cout << "compute the number of diamonds" << '\n';
  diamondSolver(g, k, total);
	double run_time = omp_get_wtime() - start_time;
	t.Stop();
  std::cout << "runtime [omp_base] = " << run_time << " sec\n";
  return;	
}

void CliqueSolver(Graph &g, int k, uint64_t &total, int, int){
	return;
}