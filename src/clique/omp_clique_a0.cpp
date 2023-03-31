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

void cnt_1(Graph &g, vector<vector<vidType>> &adj, vector<vidType> &type, vidType &n, uint64_t &t1){
  // counts all HHHH 
	vector<vidType> h;
  vector<vidType> label_to_h(n);
  for (vidType i = 0; i < n; i++){
    if (type[i]==1){
      h.push_back(i);
      label_to_h[i] = h.size()-1;
    }
  }
  vidType m = h.size();

	vector<vector<vidType>> adj_h(m);
	for (vidType i = 0; i < m; i++){
		for (vidType j : adj[h[i]]){
			if (type[j]==1){
				adj_h[i].push_back(label_to_h[j]);
				adj_h[label_to_h[j]].push_back(i);
			}
		}
	}

  vector<vector<vidType>> adj_arr_h(m, vector<vidType>(m));
  vector<vector<vidType>> adj_arr_h_2=adj_arr_h;
	
	for (vidType i = 0; i < m; i++){
		cout << i << endl;

		vidType mh = adj_h[i].size();
		vector<vector<vidType>> adj_arr_h_i(mh, vector<vidType>(mh));
		vector<vector<vidType>> adj_arr_h_i_2 = adj_arr_h_i;
		// fill with N(i)
		for (vidType j : adj_h[i]){
			for (vidType k : adj_h[i]){
				if (j==k) continue;
				adj_arr_h_i[j][k]=1;
			}
		}
		MM(adj_arr_h_i_2, adj_arr_h_i);
		
		// each triangle is triple counted
		for (vidType j = 0; j < mh; j++){
			for (vidType k = j+1; k < mh; k++){
				t1 += adj_arr_h_i[j][k]*adj_arr_h_i_2[j][k];
			}
		}
	}
	cout << t1 << endl;
	// 4 triangles / k4, 3 cnts / triangle
	t1 /= 12;
}

void cliqueSolver(Graph &g, int k, uint64_t &total){
	assert(k==4);
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
	uint64_t t1=0, t2=0, t3=0, t4=0, t5=0;
	cnt_1(g, adj, type, n, t1);
	total = t1+t2+t3+t4+t5; 
}

void CliqueSolver(Graph &g, int k, uint64_t &total, int, int){
	int num_threads = 1;
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	std::cout << "OpenMP " << k << "-clique listing (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  double start_time = omp_get_wtime();
	cout << "compute the number of " << k << "-cliques" << '\n';
  cliqueSolver(g, k, total);
	double run_time = omp_get_wtime() - start_time;
	t.Stop();
  std::cout << "runtime [omp_base] = " << run_time << " sec\n";
  return;	
}

void DiamondSolver(Graph &g, int k, uint64_t &total, int, int){
	return;
}
