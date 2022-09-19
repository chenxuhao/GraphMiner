#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <cstring>
#include <sstream>
#include <queue>
#include <algorithm>
#include <sys/time.h> 	//to measure the execution time
#include <omp.h>
#include <list>
using namespace std;

typedef struct Node
{
	int id;
	set<int> neighbour;
} Node;

typedef struct idOrder
{
	bool bigger;
	int ancestor_pos;
} idOrder;

typedef vector<Node> Graph;
typedef vector<vector<idOrder>> PrtlOrder;
typedef vector<vector<int>> MtchOrder;

Graph g;
int PATTERN_SIZE;
MtchOrder MO;
PrtlOrder PO;

void print_pattern (vector<int>& ancestors)
{
	cout << endl;
	cout << "Printing nodes in the pattern " << endl;
	for (int v_id : ancestors)
		cout << v_id << " ";	
}

void print_pattern (vector<int>& ancestors, int u)
{
	print_pattern(ancestors);
	cout << u;
}

void define_pattern ()
{
	PATTERN_SIZE = 3;
	MO = MtchOrder(PATTERN_SIZE);
	PO = PrtlOrder(PATTERN_SIZE);

	MO[0] = vector<int>{};
	MO[1] = vector<int>{0};
	MO[2] = vector<int>{0, 1};

	PO[0] = vector<idOrder>{};
	PO[1] = vector<idOrder>{{true, 0}};
	PO[2] = vector<idOrder>{{true, 1}};
}

void input_graph ()
{
	cout << "Enter the number of nodes: ";
	int n; //Number of nodes in the graph
	cin >> n;
	g = Graph(n);
	Node v;
	for (int i = 0; i < n; ++i)
	{
		cout << "Enter the neighbours of node " << i << ": ";
		cin.clear();
		bool ignore = true;
		v.id = i;
		while (ignore or cin.peek() != '\n')
		{
			ignore = false;
			int nghb_id;
			cin >> nghb_id;
			v.neighbour.insert(nghb_id);
		}
		g[i] = v;
		v.neighbour.clear();
	}
}

void init ()
{
	define_pattern();
	input_graph();
}

//TODO: result by reference instead of by value
set<int> neighbour_intersection (vector<int>& ancestors, vector<int>& cnct_nodes, int begin, int end)
{
	int n_intersection = end - begin;
	if (n_intersection == 1)
	{
		int ancestor_id = ancestors[cnct_nodes[begin]]; 
		return g[ancestor_id].neighbour;
	}
	set<int> intersect1;
	set<int> intersect2;
	set<int> intersect;
	#pragma omp task shared(intersect1)
	intersect1 = neighbour_intersection(ancestors, cnct_nodes, begin, begin + n_intersection/2);
	#pragma omp task shared(intersect2)
	intersect2 = neighbour_intersection(ancestors, cnct_nodes, begin + n_intersection/2, end);	
	#pragma omp taskwait

	set_intersection(intersect1.begin(), intersect1.end(), 
		             intersect2.begin(), intersect2.end(),
		             inserter(intersect, intersect.begin()));

	return intersect;
}

//Returns true if it complies with the rules of the partial order
bool prtl_order_ok (int v_id, vector<int>& ancestors, vector<idOrder>& prtl_order)
{
	bool prtl_ok = true;
	#pragma omp taskloop shared(prtl_ok)
	for (int i = 0; i < prtl_order.size(); ++i)
	{
		int anc_pos = prtl_order[i].ancestor_pos; 
		if (anc_pos < ancestors.size())	//It could be making a reference to a future ancestor
			if ((prtl_order[i].bigger) ? v_id <= ancestors[anc_pos] : v_id >= ancestors[anc_pos])
				prtl_ok = false; 
	}
	return prtl_ok;
}

//cnct_nodes has at least one element
//TODO: result by reference instead of by value
vector<int> generate_frontier (vector<int>& ancestors, vector<int>& cnct_nodes, vector<idOrder>& prtl_order)
{
	if (ancestors.empty())
		return vector<int>(0);
	set<int> nghb_intersect = neighbour_intersection(ancestors, cnct_nodes, 0, cnct_nodes.size());
	
	//Cast into a vector so the for loop can be parallel
	int n_neigh = nghb_intersect.size();
	vector<int> v_nghb_intersect(n_neigh);
	copy(nghb_intersect.begin(), nghb_intersect.end(), v_nghb_intersect.begin());
	vector<int> frontier; 
	if (n_neigh > 0) //If not it gives segmentation fault
	{
		#pragma omp declare reduction (merge : vector<int> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
		#pragma omp taskloop reduction(merge:frontier)
		for (int i = 0; i < n_neigh; ++i)
			if (prtl_order_ok(v_nghb_intersect[i], ancestors, prtl_order))
				frontier.push_back(v_nghb_intersect[i]);
	}
	return frontier;
}

void dfs_search_internal (int v_id, vector<int>& ancestors)
{
	ancestors.push_back(v_id);
	int depth = ancestors.size();
	vector<int> frontier = generate_frontier(ancestors, MO[depth], PO[depth]);
	for (int u : frontier)
		if (depth == PATTERN_SIZE - 1)
			print_pattern(ancestors, u);
		else 
			dfs_search_internal(u, ancestors);

	ancestors.pop_back();
}

void dfs_search ()
{
	#pragma omp taskloop
	for (int i = 0; i < g.size(); ++i)
	{
		vector<int> ancestors;
		dfs_search_internal(g[i].id, ancestors);
	}
}

//All vectors in v_ancestors represent a different path of the same depth
void bfs_search_internal (vector<vector<int>>& v_ancestors)
{
	if (v_ancestors.size() == 0)	
		return;
	
	int depth = v_ancestors[0].size();
	if (depth == PATTERN_SIZE)
	{
		for(int i = 0; i < v_ancestors.size(); ++i)
			print_pattern(v_ancestors[i]);
		return;
	}

	vector<vector<int>> nxt_anc;
	#pragma omp declare reduction (merge : vector<vector<int>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
	#pragma omp taskloop reduction(merge:nxt_anc)
	for(int i = 0; i < v_ancestors.size(); ++i)
	{
		vector<int> ancestors = v_ancestors[i];
		vector<int> frontier = generate_frontier(ancestors, MO[depth], PO[depth]);
		vector<vector<int>> nxt_anc_private(frontier.size(), ancestors);
		
		#pragma omp taskloop shared(nxt_anc_private)
		for (int j = 0; j < frontier.size(); ++j)
		{
			nxt_anc_private[j].push_back(frontier[j]);
		}
		nxt_anc.insert(nxt_anc.end(), nxt_anc_private.begin(), nxt_anc_private.end());
	}
	bfs_search_internal(nxt_anc);
}


void bfs_search ()
{
	#pragma omp taskloop
	for (int i = 0; i < g.size(); ++i)
	{
		vector<vector<int>> v_ancestors {{g[i].id}};
		bfs_search_internal(v_ancestors);
	}
}

int main(int argc, char *argv[])
{
	init();
	struct timeval start, end;
	gettimeofday(&start, NULL); //Start timing the computation

	//Set the number of threads
	omp_set_dynamic(0);     	// Explicitly disable dynamic teams 
	omp_set_num_threads(4);
	if (argc > 1 and !strcmp(argv[1], "bfs"))
	{
		#pragma omp parallel
		#pragma omp single
		bfs_search();
		cout << endl;
		cout << "search done with bfs" << endl;
	}
	else
	{
		#pragma omp parallel
		#pragma omp single
		dfs_search();
		cout << endl;
		cout << "search done with dfs" << endl;
	}
	gettimeofday(&end, NULL); //Stop timing the computation

	double myTime = (end.tv_sec+(double)end.tv_usec/1000000) - (start.tv_sec+(double)start.tv_usec/1000000);
	cout << "Execution time: " << myTime << " seconds." << endl;
}