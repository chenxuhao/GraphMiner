#include "graph.h"

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

void cnt_3(Graph &g, vector<vector<vidType>> &adj, vector<vidType> &type, vidType &n, uint64_t &total_3){
	// counts all HH diameters with at least one L other
	// O((e/D)^2*e)?
	vector<vidType> h;
	vector<vidType> label_to_h(n);
	for (vidType i = 0; i < n; i++){
		if (type[i]==1){
			h.push_back(i);
			label_to_h[i] = (vidType)h.size()-1;
		}
	}

	vidType m = h.size();
	vector<vector<vidType>> adj_arr_h(m, vector<vidType>(m));
  for (vidType i = 0; i < m; i++){
    for (vidType j : adj[h[i]]){
      if (type[j]==1){
        adj_arr_h[i][label_to_h[j]] = 1;
        adj_arr_h[label_to_h[j]][i] = 1;
      }
    }
  }

	for (vidType i = 0; i < m; i++){
		for (vidType j = 0; j < m; j++){
			if (!adj_arr_h[i][j]) continue;
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
	total_3 /= 2;
}

void cnt_2(Graph &g, vector<vector<vidType>> &adj, vector<vidType> &type, vidType &n, uint64_t &total_2){
	// counts all HL and LL diameters with any others
	// O(sum deg_L^alpha)\in D^(alpha-1)*e, summing over adjacency graphs of all low degree vertices
	for (vidType i = 0; i < n; i++){
		if (type[i]==0){
			// cout << "found a low degree vertex, with degree: " << (vidType)adj[i].size() << '\n';
			vector<vidType> nx;
			vector<vidType> label_to_nx(n);
			nx.push_back(i);
			label_to_nx[i] = 0;
			for (auto u : adj[i]){
				nx.push_back(u);
				label_to_nx[u] = nx.size()-1;
			}
			vidType m = nx.size();
			vector<vector<vidType>> adj_x(m, vector<vidType>(m));
			for (vidType i = 0; i < m; i++){
				for (vidType j : adj[nx[i]]){
					adj_x[i][label_to_nx[j]] = 1;
					adj_x[label_to_nx[j]][i] = 1;
				}
			}
			
			vector<vector<vidType>> adj_x_2=adj_x;
			MM(adj_x_2, adj_x);
			for (auto u : adj[i]){
				if (type[u] == 1){ // HL
					total_2 += adj_x_2[label_to_nx[i]][label_to_nx[u]]*(adj_x_2[label_to_nx[i]][label_to_nx[u]]-1);
				} else { // LL
					total_2 += adj_x_2[label_to_nx[i]][label_to_nx[u]]*(adj_x_2[label_to_nx[i]][label_to_nx[u]]-1)/2;
				}
			}
			// cout << "new total 2 value: " << total_2 << '\n';
		}
	}
	total_2/=2;	
}

void cnt_1(Graph &g, vector<vector<vidType>> &adj, vector<vidType> &type, vidType &n, uint64_t &total_1){
  // counts all HH diameters with HH others
	// O(sum deg_H^alpha)\in O(deg_H*sum deg_H^(alpha-1))\in O(e^alpha/D^(alpha-1))	
	vector<vidType> h;
	vector<vidType> label_to_h(n);
  for (vidType i = 0; i < n; i++){
    if (type[i]==1){
      h.push_back(i);
      label_to_h[i] = h.size()-1;
    }
  }
	vidType m = h.size();

	vector<vector<vidType>> adj_arr_h(m, vector<vidType>(m));
	for (vidType i = 0; i < m; i++){
		for (vidType j : adj[h[i]]){
			if (type[j]==1){
				adj_arr_h[i][label_to_h[j]] = 1;
				adj_arr_h[label_to_h[j]][i] = 1;
			}
		}
	}
	
	vector<vector<vidType>> adj_arr_h_2=adj_arr_h;
	MM(adj_arr_h_2, adj_arr_h);

	for (vidType i = 0; i < m; i++){
		for (vidType j = 0; j < m; j++){
			if (adj_arr_h_2[i][j] && adj_arr_h[i][j]){
				total_1 += adj_arr_h_2[i][j]*(adj_arr_h_2[i][j]-1)/2;
			}	
		}
	}
	total_1/=2;
}

void diamondSolver(Graph &g, int k, uint64_t &total){
	// assert(k==4);
	vidType n = g.V();
	vector<vector<vidType>> adj(n);
  for (vidType i = 0; i < n; i++){
    auto ni = g.N(i);
    for (auto u : ni){
      adj[i].push_back(u);
    }
  }

	vidType D = pow(g.E(), 0.5); // TODO: what is the best possible threshold? 
	D=10; // for testing
	cout << "number of edges: " << g.E() << '\n';
	cout << "threshold for high degree vertices: " << D << '\n';
	vidType num_high_deg = 0;
	vector<vidType> type(n);
	for (vidType i = 0; i < n; i++){
		if ((vidType)adj[i].size() >= D) {type[i] = 1; num_high_deg++;}
	}
	cout << "number of high degree vertices: " << num_high_deg << '\n';
	// totals
	uint64_t total_1 = 0, total_2 = 0, total_3 = 0;
	cnt_1(g, adj, type, n, total_1);
	cnt_2(g, adj, type, n, total_2);
	cnt_3(g, adj, type, n, total_3);
	total = total_1 + total_2 + total_3; 
	cout << "HHHH total: " << total_1 << '\n';
	cout << "HL/LL diameter total: " << total_2 << '\n';
	cout << "HH diamater with>=1 L total: " << total_3 << '\n';
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
