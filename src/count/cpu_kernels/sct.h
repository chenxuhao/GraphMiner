// #pragma omp parallel for schedule(dynamic,1) reduction(+:counter)
enum CallType { PIVOT, HOLD };
vector<int> position = g.degeneracy_ordering();

vector<vector<uint64_t> > C = {{1}};
for (int i = 1; i <= 100; i++) {
  vector<uint64_t> new_row = {1};
  for (int j = 0; j < i - 1; j++) new_row.push_back(C.back()[j] + C.back()[j + 1]);
  new_row.push_back(1);
  C.push_back(new_row);
}

struct SCT {
  VertexList vlabel;
  pair<vidType, CallType> elabel;
  vector<SCT*> children;
  void count_cliques(vector<int>& p, vector<int>& h, vector<int>& cnt, vector<vector<uint64_t> >& C) {
    if (elabel.second == PIVOT) p.push_back(elabel.first);
    else h.push_back(elabel.first);
    vidType p_sz = p.size();
    vidType h_sz = h.size();
    if (children.empty()) {
      for (int i = 0; i < p_sz; i++) cnt[h_sz + i] += C[p_sz - 1][i];
    }
    else {
      for (SCT* child: children) child->count_cliques(p, h, cnt, C);
    }
    if (elabel.second == PIVOT) p.pop_back();
    else h.pop_back();
  };
};

SCT* sct = new SCT();
sct->elabel = {-1, PIVOT};
for (vidType v = 0; v < g.size(); v++) sct->vlabel.push_back(v);

queue<SCT*> q;
for (vidType v = 0; v < g.size(); v++) {
  SCT* child = new SCT();
  for (vidType u: g.N(v)) {
    if (position[u] > position[v]) child->vlabel.append(u);
  }
  // copy(g.out_colidx() + g.edge_begin(v), g.out_colidx() + g.edge_end(v), back_inserter(child->vlabel));
  child->elabel = {v, HOLD};
  sct->children.push_back(child);
  q.push(child);
}

while (!q.empty()) {
  SCT* node = q.front();
  q.pop();
  if (node->vlabel.size() == 0) continue;

  // get all N(S, v)
  map<vidType, VertexList> NS;
  for (vidType v: node->vlabel) {
    set_intersection(
      g.out_colidx() + g.edge_begin(v), g.out_colidx() + g.edge_end(v),
      begin(node->vlabel), end(node->vlabel),
      back_inserter(NS[v])
    );
    // VertexList tmp;
    // set_union(
    //   g.out_colidx() + g.out_rowptr()[v], g.out_colidx() + g.out_rowptr()[v + 1],
    //   g.in_colidx() + g.in_rowptr()[v], g.in_colidx() + g.in_rowptr()[v + 1],
    //   back_inserter(tmp)
    // );
    // set_intersection(
    //   begin(tmp), end(tmp),
    //   begin(node->vlabel), end(node->vlabel),
    //   back_inserter(NS[v])
    // );
  }
  // find the pivot
  vidType p = node->vlabel[0];
  for (vidType v: node->vlabel) {
    if (NS[v].size() > NS[p].size()) p = v;
  }
  SCT* child = new SCT();
  child->vlabel = NS[p];
  child->elabel = {p, PIVOT};
  node->children.push_back(child);
  q.push(child);

  // find other children
  VertexList others;
  VertexList x, y = {p};
  set_union(
    begin(NS[p]), end(NS[p]),
    begin(y), end(y),
    back_inserter(x)
  );
  set_difference(
    begin(node->vlabel), end(node->vlabel),
    begin(x), end(x),
    back_inserter(others)
  );
  while (!others.empty()) {
    vidType v = others.back();
    others.pop_back();
    SCT* child = new SCT();
    set_difference(
      begin(NS[v]), end(NS[v]),
      begin(others), end(others),
      back_inserter(child->vlabel)
    );
    child->elabel = {v, HOLD};
    node->children.push_back(child);
    q.push(child);
  }
}

vector<int> cnt(g.size() + 1);

vector<int> p, h;
sct->count_cliques(p, h, cnt, C);

int idx = 1;
while (idx < cnt.size() && cnt[idx]) cout << cnt[idx++] << " ";