#pragma once

class Pattern {
public:
  Pattern() {};
  ~Pattern() {};
  void init(int n, int m, int* srcs, int* dsts) {
    num_verts = n;
    num_edgs = 2*m;
    max_deg = 0;
    adjacency_array = new int[2*m];
    degree_list = new int[n+1];
    degree_list[0] = 0;
    int* temp_deg_list = new int[n];
    for (int i = 0; i < n; ++i)
      temp_deg_list[i] = 0;
    for (int i = 0; i < m; ++i) {
      temp_deg_list[srcs[i]]++;
      temp_deg_list[dsts[i]]++;
    }
    for (int i = 0; i < n; ++i)
      max_deg = temp_deg_list[i] > max_deg ? temp_deg_list[i] : max_deg;
    for (int i = 0; i < n; ++i)
      degree_list[i+1] = degree_list[i] + temp_deg_list[i];
    copy(degree_list, degree_list+n, temp_deg_list);
    for (int i = 0; i < m; ++i) {
      adjacency_array[temp_deg_list[srcs[i]]++] = dsts[i];
      adjacency_array[temp_deg_list[dsts[i]]++] = srcs[i];
    }
    delete [] temp_deg_list;
  }
  int* adjacent_vertices(int v) {
    return (&adjacency_array[degree_list[v]]);
  }
  int out_degree(int v) {
    return degree_list[v+1] - degree_list[v];
  }
  int* adjacencies() const {
    return adjacency_array;
  }
  int* degrees() const {
    return degree_list;
  }
  int num_vertices() const {
    return num_verts;
  }
  int num_edges() const {
    return num_edgs;
  }
  int max_degree() const {
    return max_deg;
  }
  Pattern& operator= (const Pattern& param) {
    num_verts = param.num_vertices();
    num_edgs = param.num_edges();
    max_deg = param.max_degree();
    adjacency_array = new int[2*num_edgs];
    degree_list = new int[num_verts+1];    
    copy(param.adjacencies(), param.adjacencies() + 2*num_edgs, adjacency_array);
    copy(param.degrees(), param.degrees() + (num_verts+1), degree_list);
    return *this;
  }
  void clear() {
    delete [] adjacency_array;
    delete [] degree_list;
  }
  
private:
  int num_verts;
  int num_edgs;
  int max_deg;
  int* adjacency_array;
  int* degree_list;
};

void read_in_pattern(Pattern& g, char* graph_file, bool labeled, int*& srcs_g, int*& dsts_g, int*& labels_g) {
  ifstream file_g;
  string line;
  file_g.open(graph_file);
  int n_g;
  int m_g;    
  getline(file_g, line);
  n_g = atoi(line.c_str());
  getline(file_g, line);
  m_g = atoi(line.c_str());
  srcs_g = new int[m_g];
  dsts_g = new int[m_g];
  if (labeled) {
    labels_g = new int[n_g];
    for (int i = 0; i < n_g; ++i) {
      getline(file_g, line);
      labels_g[i] = atoi(line.c_str());
    }
  } else {
    labels_g = NULL;
  }
  for (int  i = 0; i < m_g; ++i) {
    getline(file_g, line, ' ');   
    srcs_g[i] = atoi(line.c_str());
    getline(file_g, line);  
    dsts_g[i] = atoi(line.c_str());
  } 
  file_g.close();
  g.init(n_g, m_g, srcs_g, dsts_g);
}

