// Copyright (c) 2013, The Pennsylvania State University.
// All rights reserved.
// 
// See COPYING for license.

using namespace std;

class output{
public:
  output(double* vc, int nv)
  { 
    vert_counts = vc;
    num_verts = nv;
  }

  output(double** vc, int num_vert_counts, int nv)
  { 
    num_verts = nv;
    vert_counts = vc[0];

    for (int v = 0; v < num_verts; ++v)
    {
      for (int i = 1; i < num_vert_counts; ++i)
        vert_counts[v] += vc[i][v];
      vert_counts[v] /= num_vert_counts;
    }
  } 

  ~output()
  { };

  void output_gdd(char* output_filename)
  {
    vector<pair<double, int> > gdd(0);
          
    for (int v = 0; v < num_verts; ++v)
    {
      double count_v = vert_counts[v];      
      int i = get_count_index(gdd, count_v);      
      gdd[i].second += 1;
    }  
    
    do_gdd_sort(gdd);
    write_gdd(output_filename, gdd);
  }

  void write_gdd(char* output_filename, vector<pair<double, int> > gdd)
  {
    ofstream out;
    out.open(output_filename);
    
    out << "Count,Nodes" << "\n";
    for (size_t i = 0; i < gdd.size(); ++i)
      out << gdd[i].first << "," << gdd[i].second << "\n";
  
    out.close();
  }

  void output_verts(char* output_filename)
  {
    ofstream out;
    out.open(output_filename);
    
    out << "VID,Count" << "\n";
    for (int i = 0; i < num_verts; ++i)
      out << i << "," << vert_counts[i] << "\n";
  
    out.close();
  }
  
private:
  int get_count_index(vector<pair<double, int> >& gdd, double count)
  {
    int gf_size = gdd.size();
    
    for (int i = 0; i < gf_size; ++i)
      if (gdd[i].first == count)
        return i;
        
    pair<double, int> new_pair;
    new_pair.first = count;
    new_pair.second = 0;
    
    gdd.push_back(new_pair);
    
    return gf_size;
  }

  void do_gdd_sort(vector<pair<double, int> >& gdd)
  {
    bool swapped;    
    do
    {
      swapped = false;
      for (size_t i = 1; i < gdd.size(); ++i)
      {
        if (gdd[i].first < gdd[i-1].first)
        {
          pair<double, int> temp;
          temp.first = gdd[i].first;
          temp.second = gdd[i].second;
          gdd[i] = gdd[i-1];
          gdd[i-1] = temp;
          swapped = true;
        }
      }
    } while (swapped);
  }

  double* vert_counts;
  int num_verts;  
};
