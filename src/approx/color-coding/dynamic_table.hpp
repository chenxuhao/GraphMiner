#pragma once
#include "pattern.hpp"
using namespace std;

class dynamic_table {
public:
  dynamic_table() {
    choose_table = NULL;
    num_colors = 0;
    num_verts = 0;
    num_subs = 0;
    is_inited = false;
    is_sub_inited = NULL;
  }
  ~dynamic_table() {
    delete [] num_colorsets;
    delete [] choose_table;
  }
  virtual void init(Pattern* subtemplate, int num_subtemplates, int num_vertices, int num_colors){};
  virtual void init_sub(int subtemplate){};
  virtual void clear_sub(int subtemplate){};
  virtual void clear_table(void){};  
  virtual bool is_init(void) = 0;
  virtual bool is_sub_init(int subtemplate) = 0;

protected:
  void init_choose_table() {      
    choose_table = new int*[num_colors + 1];
    for (int i = 0; i <= num_colors; ++i)
      choose_table[i] = new int[num_colors + 1];
    for (int i = 0; i <= num_colors; ++i)
      for (int j = 0; j <= num_colors; ++j)
        choose_table[i][j] = choose(i, j);
  }
  void init_num_colorsets() {
    num_colorsets = new int[num_subs];  
    for (int s = 0; s < num_subs; ++s)
      num_colorsets[s] = choose(num_colors, subtemplates[s].num_vertices());
  }

  Pattern* subtemplates;
  int** choose_table;
  int* num_colorsets;
  int num_colors;
  int num_subs;
  int num_verts;
  bool is_inited;
  bool* is_sub_inited;
};
