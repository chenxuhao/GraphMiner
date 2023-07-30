#pragma once
#include "pattern.hpp"
using namespace std;

class partitioner {
public:
  partitioner()  {}  
  partitioner(Pattern& t, bool label, int* label_map) {
    init_arrays();
    labeled = label;
    subtemplates_create[0] = t;
    current_creation_index = 1;  

    if (labeled)
      label_maps.push_back(label_map);
    parents.push_back((long)NULL_VAL);    

    int root = 0;
    partition_recursive(0, root);

    fin_arrays();
  }
  ~partitioner()
  {  }
  
  void clear_temparrays() {
    delete [] subtemplates;
    delete [] count_needed;
  } 

  // Do a bubble sort based on each subtemplate's parents' index
  // This is a simple way to organize the partition tree for use
  //  with dt.init_sub() and memory management of dynamic table
  void sort_subtemplates() {
    bool swapped;
    Pattern temp_g;
    
    do
    {
      swapped = false;
      for (int i = 2; i < subtemplate_count; ++i)
      {
        if (parents[i] < parents[i - 1])
        {
          temp_g = subtemplates[i];
          int temp_pr = parents[i];
          int temp_a = active_children[i];
          int temp_p = passive_children[i];
          int* temp_l = NULL;
          if (labeled)
            temp_l = label_maps[i];
        
          // if either have children, need to update their parents
          if (active_children[i] != NULL_VAL)
            parents[active_children[i]] = (i-1);          
          if (active_children[i-1] != NULL_VAL)
            parents[active_children[i-1]] = i;          
          
          if (passive_children[i] != NULL_VAL)
            parents[passive_children[i]] = (i-1);
          if (passive_children[i-1] != NULL_VAL)
            parents[passive_children[i-1]] = i;
            
          // need to update their parents
          if (active_children[parents[i]] == i)
            active_children[parents[i]] = (i-1);
          else if (passive_children[parents[i]] == i)
            passive_children[parents[i]] = (i-1);        
          
          if (active_children[parents[i-1]] == (i-1))
            active_children[parents[i-1]] = i;
          else if (passive_children[parents[i-1]] == (i-1))
            passive_children[parents[i-1]] = i;
          
          // make the switch
          subtemplates[i] = subtemplates[i-1];
          parents[i] = parents[i-1];
          active_children[i] = active_children[i-1];
          passive_children[i] = passive_children[i-1];
          if (labeled)
            label_maps[i] = label_maps[i-1];
          
          subtemplates[i-1] = temp_g;
          parents[i-1] = temp_pr;
          active_children[i-1] = temp_a;
          passive_children[i-1] = temp_p;
          if (labeled)
            label_maps[i-1] = temp_l;
          
          swapped = true;
        }
      }
    } while (swapped);    
  }
  
  int sub_count_needed(int s) { return count_needed[s]; }  
  
  Pattern* get_subtemplates() { return subtemplates; }
  
  int get_subtemplate_count() { return subtemplate_count; }
  
  int* get_labels(int s)
  {
    if (labeled)
      return label_maps[s];
    else
      return NULL;
  }
  
  int get_active_index(int a)
  {
    return active_children[a];
  }  
  
  int get_passive_index(int p)
  {
    return passive_children[p];
  }
  
  int get_num_verts_active(int s)
  {
    return subtemplates[active_children[s]].num_vertices();
  }
  
  int get_num_verts_passive(int s)
  {
    return subtemplates[passive_children[s]].num_vertices();
  }
    
  
private:  
  // Initialize dynamic arrays and graph array
  void init_arrays()
  {
    subtemplates_create = new Pattern[CREATE_SIZE];
            
    parents = vector<int>(0);
    active_children = vector<int>(0);
    passive_children = vector<int>(0);
    label_maps = vector<int*>(0);
  }
  
  // Finalize arrays
  // Delete the creation array and make a final one of appropriate size
  void fin_arrays()
  {
    subtemplate_count = current_creation_index;    
    subtemplates = new Pattern[subtemplate_count];
    
    for (int i = 0; i < subtemplate_count; ++i)
    {
      subtemplates[i] = subtemplates_create[i];
      subtemplates_create[i].clear();
    }    
    delete [] subtemplates_create;
    
    count_needed = new bool[subtemplate_count];
    for (int i = 0; i < subtemplate_count; ++i)
      count_needed[i] = true;
  }
  

  void partition_recursive(int s, int root)
  {
    // split the current subtemplate using the current root
    int* roots = split(s, root);
    
    // set the parent/child tree structure
    int a = current_creation_index - 2;
    int p = current_creation_index - 1;          
    set_active_child(s, a);
    set_passive_child(s, p);
    set_parent(a, s);
    set_parent(p, s);

    //specify new roots and continue partitioning
    int num_verts_a = subtemplates_create[a].num_vertices();
    int num_verts_p = subtemplates_create[p].num_vertices();        

    if (num_verts_a > 1) 
    {
      int activeRoot = roots[0];
      partition_recursive(a, activeRoot);
    }
    else 
    {
      set_active_child(a, NULL_VAL);
      set_passive_child(a, NULL_VAL);
    }
    
    if (num_verts_p > 1) 
    {
      int passiveRoot = roots[1];
      partition_recursive(p, passiveRoot);
    }
    else 
    {
      set_active_child(p, NULL_VAL);
      set_passive_child(p, NULL_VAL);
    }      
  }

  int* split(int& s, int& root)
  {
    // get new root
    int* adjs = subtemplates_create[s].adjacent_vertices(root);  
    int u = adjs[0];

    // split this subtemplate between root and node u
    int active_root = split_sub(s, root, u);
    int passive_root = split_sub(s, u, root);
    
    int* ret = new int[2];
    ret[0] = active_root;
    ret[1] = passive_root;
    return ret;
  }

  int split_sub(int& s, int& root, int& other_root)
  {    
    subtemplate = subtemplates_create[s];
    int* labels_sub = NULL;
    if (labeled)
      labels_sub = label_maps[s];    
    
    // source and destination arrays for edges
    vector<int> srcs;
    vector<int> dsts;
    
    // get the previous vertext to avoid backtracking
    int previous_vertex = other_root;
    
    // loop through the rest of the vertices
    // if a new edge is found, add it
    vector<int> next;
    next.push_back(root);
    size_t cur_next = 0;
    while (cur_next < next.size())
    {
      int u = next[cur_next++];
      int* adjs = subtemplate.adjacent_vertices(u);
      int end = subtemplate.out_degree(u);
      
      for (int i = 0; i < end; i++)
      {
        int v = adjs[i];
        
        bool add_edge = true;
        for (size_t j = 0; j < dsts.size(); j++)
        {
          if(srcs[j] == v && dsts[j] == u)
          {
            add_edge = false;
            break;
          }
        }
        
        if (add_edge && v != previous_vertex)
        {
          srcs.push_back(u);
          dsts.push_back(v);
          next.push_back(v);
        }
      }      
    }
    
    // if empty, just add the single vert
    int n;
    int m;  
    int* labels = NULL;
    
    if (srcs.size() > 0)
    {
      m = srcs.size();
      n = m + 1;
      
      if (labeled)
      {
        vector<int> label_ids;
        extract_uniques(next, label_ids);
        labels = new int[label_ids.size()];
      }
           
      check_nums(root, srcs, dsts, labels, labels_sub);
    }
    else
    {
      // single node
      m = 0;
      n = 1;
      srcs.push_back(0);  

      if (labeled)
      {    
        labels = new int[1];
        labels[0] = labels_sub[root];
      }
    }
    
    int* srcs_array = dynamic_to_static(srcs);
    int* dsts_array = dynamic_to_static(dsts);
    
    subtemplates_create[current_creation_index].init(n, m, srcs_array, dsts_array);
    current_creation_index++;
    
    if (labeled)
      label_maps.push_back(labels);    
    
    delete [] dsts_array;
    delete [] srcs_array;
    
    return srcs[0];
  }

  // Check nums 'closes the gaps' in the srcs and dsts arrays
  // Can't initialize a graph with edges (0,2),(0,4),(2,5)
  // This changes the edges to (0,1),(0,2),(1,3)
  void check_nums(int root, vector<int>& srcs, vector<int>& dsts, 
    int* labels, int* labels_sub)
  {
    int maximum = get_max(srcs, dsts);    
    int size = srcs.size();

    int* mappings = new int[maximum+1];
    for (int i = 0; i < maximum+1; ++i)
      mappings[i] = -1;

    int new_map = 0;
    mappings[root] = new_map++;

    for (int i = 0; i < size; ++i)
    {
      if (mappings[srcs[i]] == -1)
        mappings[srcs[i]] = new_map++;
      if (mappings[dsts[i]] == -1)
        mappings[dsts[i]] = new_map++;
    }
    for (int i = 0; i < size; ++i)
    {
      srcs[i] = mappings[srcs[i]];
      dsts[i] = mappings[dsts[i]];
    }
    if (labeled)
    {
      for (int i = 0; i < maximum; ++i)
        if (mappings[i] != -1)
          labels[mappings[i]] = labels_sub[i];
    }
  }
  void set_active_child(int s, int a) {
    while (active_children.size() <= ((size_t) s))
      active_children.push_back(NULL_VAL);
      
    active_children[s] = a;
  }
  void set_passive_child(int s, int p) {
    while (passive_children.size() <= ((size_t) s))
      passive_children.push_back(NULL_VAL);
      
    passive_children[s] = p;
  }
  void set_parent(int c, int p) {
    while (parents.size() <= ((size_t) c))
      parents.push_back(NULL_VAL);
      
    parents[c] = p;
  }
  
  Pattern* subtemplates_create;
  Pattern* subtemplates;
  Pattern subtemplate;
  vector<int> active_children;
  vector<int> passive_children;
  vector<int> parents;
  vector<int> cut_edge_labels;
  vector<int*> label_maps;
  int current_creation_index;
  int subtemplate_count;
  bool* count_needed;
  bool labeled;
};






