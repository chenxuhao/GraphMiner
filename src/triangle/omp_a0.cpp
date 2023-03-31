//#pragma GCC optimize("O3,unroll-loops")
//#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include "graph.h"
#define ll uint64_t 

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

ll intersect(vidType &x, vidType &y, vector<vidType> &A, vector<vidType> &B, vector<vidType> &type){
	// size of A < size of B
	ll cnt=0;
	vidType Asz = (vidType)A.size();
	vidType Bsz = (vidType)B.size();
	vidType idx1 = 0;
	vidType idx2 = 0;
	vidType h_cnt = type[x]+type[y];
	//for (vidType i = 0; i < Asz; i++){
	//	if (idx2 == Bsz) break;
	//	if (A[i] > B[idx2]){
	//		idx2 = lower_bound(B.begin()+idx2+1, B.end(), A[i])-B.begin();
	//		if (idx2 != Bsz && B[idx2]==A[i]){
	//			vidType h_tot = h_cnt+type[A[i]];
	//			if (h_tot == 0) cnt+=2;
	//			else if (h_tot == 1) cnt+=3;
	//			else cnt+=6;
	//			idx2++;
	//		}
	//	}
	//}
	while(idx1 != Asz && idx2 != Bsz){
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

void TCSolver(Graph &g, uint64_t &total, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP TC (" << num_threads << " threads)\n";
	Timer t;
	t.Start();
	ll counter = 0;
	vidType n = g.V();
	
	Timer p;
	p.Start();
	// convert to undirected
	//vector<vector<vidType>> adj_arr(n, vector<vidType>(n)); 
	//for (vidType i = 0; i < n; i++){
	//	auto ni = g.N(i);
	//	for (auto u : ni){
	//		adj_arr[i][u]=1;
	//		adj_arr[u][i]=1;
	//	}
	//}

	// build adjacency lists
	vector<vector<vidType>> adj(n);
	for (vidType i = 0; i < n; i++){
		auto ni = g.N(i);
		for (auto u : ni){
			adj[i].push_back(u);
		}
	}
	//for (vidType i = 0; i < n; i++){
	//	for (vidType j = i+1; j < n; j++){
	//		if (adj_arr[i][j]==1){
	//			adj[i].push_back(j);
	//			adj[j].push_back(i);
	//		}
			//for (auto u : g.N(i)){
			//	adj[i].push_back(u);
			//}
	//	}
	//}
	

	// sort adjacency lists
	vidType max_undirected_degree = 0;
	for (vidType i = 0; i < n; i++){
	//	sort(adj[i].begin(), adj[i].end());
		max_undirected_degree = max(max_undirected_degree, (vidType)adj[i].size());
	}
	cout << "max undirected degree: " << max_undirected_degree << endl;
	
	p.Stop();
	cout << "preprocessing runtime = " << p.Seconds() << " sec\n";
	Timer s;
	s.Start();
	// adjacency squared
	// vector<vector<vidType>> adj_arr_2=adj_arr;
	// MM(adj_arr_2, adj_arr);

	// collect vertices of high degree
	double alpha = 2.81; // coefficient of matrix multiplication
	vidType D = pow(g.E(),(alpha-1.0)/(alpha+1.0)); // threshold	
	// D = 700;
	cout << "threshold degree: " << D << endl;
	vector<vidType> h;
	vector<vidType> label_to_h(n);
	vector<vidType> type(n);
	for (vidType i = 0; i < n; i++){
		if ((vidType)adj[i].size() >= D){
			h.push_back(i);
			label_to_h[i] = h.size()-1;
			type[i]=1;
		}
	}

	// triangles with at least one low degree
	ll lowcounter = 0;
	#pragma omp parallel for reduction(+ : lowcounter) schedule(dynamic, 1)
	for (vidType i = 0; i < n; i++){
		if (type[i]!=1){
			for (vidType j : adj[i]){
				lowcounter += intersect(i,j,adj[i], adj[j], type);
			}
		}
	}

	// triangles with all high degrees
	ll highcounter = 0;
	vidType m = h.size();
	cout << "m: " << m << endl;
	vector<vector<vidType>> adj_arr_h(m, vector<vidType>(m));
	for (vidType i = 0; i < m; i++){
		for (vidType j : adj[h[i]]){
			if (type[j]==1){
				adj_arr_h[i][label_to_h[j]]=1;
				adj_arr_h[label_to_h[j]][i]=1;
			}
		}
	}

	vector<vector<vidType>> adj_arr_h_2=adj_arr_h;
	MM(adj_arr_h_2, adj_arr_h);

	for (vidType i = 0; i < m; i++){
		for (vidType j = 0; j < m; j++){
			highcounter += adj_arr_h_2[i][j] * adj_arr_h[i][j];
		}
	}

	// compute final answer
	cout << "lowcnt: " << lowcounter << endl;
	cout << "highcnt: " << highcounter << endl;
	total = lowcounter/12 + highcounter/6;
	s.Stop();
	t.Stop();
	cout << "computing runtime = " << s.Seconds() << " sec\n";
	std::cout << "total runtime [omp_a0] = " << t.Seconds() << " sec\n";
	return;
}


