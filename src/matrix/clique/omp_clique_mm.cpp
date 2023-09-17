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

void cnt_1(Graph &g, vector<vector<vidType>> &adj, vector<vector<vidType>> &adj_arr, vector<vidType> &type, vidType &n, uint64_t &t1){
  // t1 = 4*HHHH
	// runtime: iterates through all high vertices. sum d_H^alpha = e * (e/D)^(alpha-1)
	vector<vidType> h;
	vector<vidType> label_to_h(n);
	vector<vidType> h_to_label(n);
	for (vidType i = 0; i < n; i++){
		if (type[i]==1){
			h.push_back(i);
			label_to_h[i] = h.size()-1;
			h_to_label[h.size()-1] = i;
		}
	}
	vidType m = h.size();
	
	vector<vector<vidType>> adj_h(m);
	for (vidType i = 0; i < m; i++){
		for (vidType j : adj[h[i]]){
			if (type[j]==1){
				adj_h[i].push_back(label_to_h[j]);
			}
		}
	}

	vector<vector<vidType>> adj_arr_h(m, vector<vidType>(m));
	vector<vector<vidType>> adj_arr_h_2=adj_arr_h;

	for (vidType i = 0; i < m; i++){
	
		vidType mh = adj_h[i].size();
		vector<vector<vidType>> adj_arr_h_i(mh, vector<vidType>(mh));
		
		//if (mh > 3){
		//	cout << "i: " << i << endl;
		//	for (vidType j = 0; j < mh; j++){
		//		cout << adj_h[i][j] << " ";
		//	}
		//	cout << endl;
		//}
		// fill adjacency array with edges between neighbors N(i)
		for (vidType j = 0; j < mh; j++){
			for (vidType k = 0; k < mh; k++){
				if (adj_arr[h_to_label[adj_h[i][j]]][h_to_label[adj_h[i][k]]]){
					adj_arr_h_i[j][k]=1;
				}
			}
		}
		vector<vector<vidType>> adj_arr_h_i_2 = adj_arr_h_i;
		MM(adj_arr_h_i_2, adj_arr_h_i);
		
		// each triangle is triple counted
		int cnt = 0;
		for (vidType j = 0; j < mh; j++){
			for (vidType k = j+1; k < mh; k++){
				cnt += adj_arr_h_i[j][k]*adj_arr_h_i_2[j][k];
				t1 += adj_arr_h_i[j][k]*adj_arr_h_i_2[j][k];
			}
		}
	}

	cout << "t1 (pre division): " << t1 << endl;
	// 3 cnts / triangle
	// t1 is HHHH * 4
	t1 /= 3;
}

void cnt_2(Graph &g, vector<vector<vidType>> &adj, vector<vector<vidType>> &adj_arr, vector<vidType> &type, vidType &n, uint64_t &t2){
	// t2 is 4*LLLL + 3*LLLH + 2*LLHH + 1*LHHH
	std::cout << "inside of count 2" << endl;
}

void cliqueSolver(Graph &g, int k, uint64_t &total, int threshold){
	std::cout << "***NOT IMPLEMENTED*** \n";
	return;
	assert(k==4);
	
	vidType n = g.V();
	vector<vector<vidType>> adj(n);
	vector<vector<vidType>> adj_arr(n, vector<vidType>(n));
	for (vidType i = 0; i < n; i++){
		auto ni = g.N(i);
		for (auto u : ni){
			adj[i].push_back(u);
			adj_arr[i][u] = 1;
		}
	}

	// vidType D = pow(g.E(), 0.5); // TODO: what is the best possible threshold? 
	// D=10; // for testing
	cout << "number of edges: " << g.E() << '\n';
	cout << "threshold for high degree vertices: " << threshold << '\n';
	vidType num_high_deg = 0;
	vector<vidType> type(n);
	for (vidType i = 0; i < n; i++){
		if ((vidType)adj[i].size() >= threshold) {type[i] = 1; num_high_deg++;}
	}
	cout << "number of high degree vertices: " << num_high_deg << '\n';
	// totals
	uint64_t t1=0, t2=0, t3=0, t4=0, t5=0;
	cnt_1(g, adj, adj_arr, type, n, t1);
	total = t1+t2+t3+t4+t5; 
}

void CliqueSolver(Graph &g, int k, uint64_t &total, int threshold, int, int){
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
	cliqueSolver(g, k, total, threshold);
	double run_time = omp_get_wtime() - start_time;
	t.Stop();
	std::cout << "runtime [omp_base] = " << run_time << " sec\n";
	return;	
}