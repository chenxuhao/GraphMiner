#include "graph_partition.h"
#include "scan.h"
#include "utils.h"

void PartitionedGraph::print_subgraphs() {
  for (int i = 0; i < num_subgraphs; ++i) {
    std::cout << "Printing subgraph[" << i << "]\n";
    subgraphs[i]->print_graph();
    std::cout << "vertex id map: ";
    //vidType local_id = 0;
    for (auto v : idx_map[i]) {
      std::cout << v << " ";
    }
    std::cout << "\n\n";
  }
}

// naive 1D partitioning, i.e., edge-cut
void PartitionedGraph::edgecut_partition1D() {
  num_subgraphs = num_vertex_chunks;
}

// Given a subset of vertices and a graph g, generate a subgraph sg from the graph g
void PartitionedGraph::generate_induced_subgraph(std::vector<int8_t> v_masks, Graph *g, Graph *subg, int subg_id) {
  //std::cout << "generating induced subgraph\n";
  auto nv = idx_map[subg_id].size(); // new graph (subgraph) size
  VertexList new_ids(g->V(), 0);
  #pragma omp parallel for
  for (size_t i = 0; i < nv; ++i) {
    auto v = idx_map[subg_id][i];
    new_ids[v] = i; // reindex
    if (v == begin_vids[subg_id]) local_begin[subg_id] = i;
    if (v == end_vids[subg_id]-1) local_end[subg_id] = i+1;
  }
  //std::cout << "Computing degrees\n";
  std::vector<vidType> degrees(nv, 0); // degrees of vertices in the subgraph
  #pragma omp parallel for
  for (size_t i = 0; i < nv; ++i) {
    auto v = idx_map[subg_id][i];
    //std::cout << v << ": ";
    for (auto u : g->N(v)) {
      //if (v_set.find(u) != v_set.end()) {
      //if (std::find(idx_map[subg_id].begin(), idx_map[subg_id].end(), u) != idx_map[subg_id].end()) {
      if (v_masks[u]) {
        //std::cout << u << " ";
        degrees[i] ++;
      }
    }
    //std::cout << "\n";
  }
  //std::cout << "degrees: \n";
  //for (auto d : degrees)
  //  std::cout << d << " ";
  //std::cout << "\n";
  //auto offsets = utils::prefix_sum(degrees);
  eidType *offsets = custom_alloc_global<eidType>(nv+1);
  parallel_prefix_sum<vidType,eidType>(degrees, offsets);
  auto ne = offsets[nv];
  std::cout << "|V| = " << nv << " |E| = " << ne << "\n";
  subg->allocateFrom(nv, ne);
  #pragma omp parallel for
  for (size_t v = 0; v < nv; v++) {
    auto begin = offsets[v];
    auto end = offsets[v+1];
    subg->fixEndEdge(v, end);
    vidType j = 0;
    auto global_id = idx_map[subg_id][v];
    for (auto u : g->N(global_id)) {
      //if (v_set.find(u) != v_set.end()) {
      //if (std::find(idx_map[subg_id].begin(), idx_map[subg_id].end(), u) != idx_map[subg_id].end()) {
      if (v_masks[u]) {
        auto w = new_ids[u];
        assert(size_t(w) < nv);
        subg->constructEdge(begin+j, w);
        j++;
      }
    }
  }
}

// edge-cut 1D partitioning; generate a vertex-induced subgraph for each partition
void PartitionedGraph::edgecut_induced_partition1D() {
  auto nv = g->V();
  num_subgraphs = num_vertex_chunks;
  int subgraph_size = nv / num_subgraphs;
  if (nv % num_subgraphs != 0) subgraph_size ++;
  std::cout << "num_subgraphs: " << num_subgraphs << " subgraph_size: " << subgraph_size << "\n";
  subgraphs.resize(num_subgraphs);
  idx_map.resize(num_subgraphs);
  begin_vids.resize(num_subgraphs);
  end_vids.resize(num_subgraphs);
  local_begin.resize(num_subgraphs);
  local_end.resize(num_subgraphs);

  //std::vector<vidType> nv_of_subgraphs(num_subgraphs, 0); // number of vertices in subgraphs
  //std::vector<eidType> ne_of_subgraphs(num_subgraphs, 0); // number of edges in subgraphs
  Timer t;
  std::set<vidType> vertex_set;
  std::vector<int8_t> vertex_masks(nv, 0);
  for (int i = 0; i < num_subgraphs; ++i) {
    t.Start();
    begin_vids[i] = subgraph_size*i;
    end_vids[i] = begin_vids[i] + subgraph_size;
    if (end_vids[i] > nv) end_vids[i] = nv;
    std::cout << "generating subgraph[" << i << "]: from vertex " << begin_vids[i] << " to " << end_vids[i] << "\n";
    #pragma omp parallel for
    for (vidType v = begin_vids[i]; v < end_vids[i]; ++ v) {
      vertex_masks[v] = 1;
      for (auto u : g->N(v)) {
        vertex_masks[u] = 1;
      }
    }
    eidType *offsets = custom_alloc_global<eidType>(nv+1);
    parallel_prefix_sum<int8_t,eidType>(vertex_masks, offsets);
    auto m = offsets[nv]; // number of vertices in subgraph[i]
    idx_map[i].resize(m);
    //std::cout << "computing id map, size = " << m << "\n";
    #pragma omp parallel for
    for (vidType v = 0; v < nv; v++) {
      if (vertex_masks[v]) {
        idx_map[i][offsets[v]] = v;
      }
    }
    subgraphs[i] = new Graph();
    //g->print_graph();
    generate_induced_subgraph(vertex_masks, g, subgraphs[i], i);
    //vertex_set.clear();
    std::fill(vertex_masks.begin(), vertex_masks.end(), 0);
    t.Stop();
    //std::cout << "Time: " << t.Seconds() << " sec \n";
  }
}

// CSR segmenting
// Yunming Zhang et. al., Making caches work for graph analytics,
// 2017 IEEE International Conference on Big Data (Big Data),
// https://ieeexplore.ieee.org/document/8257937
void PartitionedGraph::csr_segmenting() {
  auto nv = g->V();
  // number of 1D partitions is the same as the number of vertex chunks (i.e. number of clusters)
  num_subgraphs = num_vertex_chunks;
  int subgraph_size = nv / num_subgraphs;
  subgraphs.resize(num_subgraphs);
  idx_map.resize(num_subgraphs);
  std::vector<int> flag(num_subgraphs, false);
  std::vector<vidType> nv_of_subgraphs(num_subgraphs, 0); // number of vertices in subgraphs
  std::vector<eidType> ne_of_subgraphs(num_subgraphs, 0); // number of edges in subgraphs
  //range_indices.resize(num_subgraphs);

  std::cout << "calculating number of vertices and edges in each subgraph\n";
  for (vidType v = 0; v < nv; ++ v) {
    for (auto u : g->N(v)) {
      auto bcol = u / subgraph_size;
      flag[bcol] = true;
      ne_of_subgraphs[bcol]++;
    }
    for (int i = 0; i < num_subgraphs; ++i) {
      if (flag[i]) nv_of_subgraphs[i] ++;
    }
    for (int i = 0; i < num_subgraphs; ++i) flag[i] = false;
  }
 
  std::cout << "allocating memory for IdxMap, RangeIndices and IntermBuf\n";
  for (int i = 0; i < num_subgraphs; ++i) {
    subgraphs[i] = new Graph();
    idx_map[i].resize(nv_of_subgraphs[i]);
    //range_indices[i].resize(num_ranges+1);
    //range_indices[i][0] = 0;
  }

  std::cout << "allocating memory for each subgraph\n";
  for (int i = 0; i < num_subgraphs; i++) {
    subgraphs[i]->allocateFrom(nv_of_subgraphs[i], ne_of_subgraphs[i]);
    ne_of_subgraphs[i] = 0;
  }

  std::cout << "constructing the blocked CSR\n";
  std::vector<int> index(num_subgraphs, 0);
  for (vidType v = 0; v < nv; ++ v) {
    for (auto u : g->N(v)) {
      int bcol = u / subgraph_size;
      subgraphs[bcol]->constructEdge(ne_of_subgraphs[bcol], u);
      flag[bcol] = true;
      ne_of_subgraphs[bcol]++;
    }
    for (int i = 0; i < num_subgraphs; ++i) {
      if (flag[i]) {
        idx_map[i][index[i]] = v; // local id (index[i]) to global id (v) mapping
        subgraphs[i]->fixEndEdge(++index[i], ne_of_subgraphs[i]);
      }
    }
    for (int i = 0; i < num_subgraphs; ++i) flag[i] = false;
  }
  std::cout << "printing subgraphs:\n";
  for (int i = 0; i < num_subgraphs; ++i) {
    subgraphs[i]->print_meta_data();
  }
}

PartitionedGraph::PartitionedGraph(Graph *g, int nc, std::vector<int> cluster_ids) :
    num_vertex_chunks(nc), num_2D_partitions(nc*nc) {
  auto nv = g->V();
  assert(cluster_ids.size() == size_t(nv)); // each vertex in g has a cluster id
  partitioned_file_path = "";
  std::cout << "num_vertex_chunks: " << num_vertex_chunks 
            << " num_2D_partitions: " << num_2D_partitions << "\n";
  verts_of_clusters.resize(nc);
  vertex_rank_in_cluster.resize(nv);
  std::fill(vertex_rank_in_cluster.begin(), vertex_rank_in_cluster.end(), 0);

  // fill vertices into clusters
  std::vector<int> nvs_of_clusters(nc, 0); // number of vertices in each cluster
  for (int v = 0; v < nv; v++) { // for each vertex v in g
    auto cid = cluster_ids[v]; // cluster id
    verts_of_clusters[cid].push_back(v);
    vertex_rank_in_cluster[v] = nvs_of_clusters[cid]++;
  }
/*
   for (int i = 0; i < num_vertex_chunks; i++) {
   std::cout << "cluster " << i << ": ";
   for (auto v : verts_of_clusters[i])
   std::cout << v << " ";
   std::cout << "\n";
   }
*/
}

// naive 2D partitioning
// partition the graph g according to the cluster id of each vertex
void PartitionedGraph::partition2D(std::vector<int> cluster_ids) {
  // degrees of vertices in each partition; degrees[i][j] is the degree of the j-th vertex in the i-th partiton
  std::vector<std::vector<int>> degrees(num_2D_partitions); 
  std::vector<std::vector<int>> rowptr_partitioned(num_2D_partitions); // row pointers of each partition
  std::vector<std::vector<int>> colidx_partitioned(num_2D_partitions); // column indices of each partition 

  // allocate memory for CSR format
  for (int i = 0; i < num_vertex_chunks; i++) {
    auto num = verts_of_clusters[i].size();
    for (int j = 0; j < num_vertex_chunks; j++) {
      auto pid = i * num_vertex_chunks + j; // partition id
      rowptr_partitioned[pid].resize(num+1);
      degrees[pid].resize(num);
      std::fill(rowptr_partitioned[pid].begin(), rowptr_partitioned[pid].end(), 0);
      std::fill(degrees[pid].begin(), degrees[pid].end(), 0);
    }
  }

  std::vector<int> nes_of_partitions(num_2D_partitions, 0); // number of edges |E| of each partition
  auto nv = g->V();
  // count the degrees and the number of edges in each partition
  for (vidType v = 0; v < nv; v++) { // for each vertex v in g
    auto src_cid = cluster_ids[v]; // src cluster id
    for (auto u : g->N(v)) { // for each neighbor u of vertex v
      auto dst_cid = cluster_ids[u]; // dst cluster id
      int pid = src_cid * num_vertex_chunks + dst_cid; // partition id
      nes_of_partitions[pid] ++; // increase the number of edges in this partition
      auto r = vertex_rank_in_cluster[v]; // v is the r-th vertex in its cluster
      degrees[pid][r]++;
    }
  }

  // computer row pointers using the degrees
  for (int i = 0; i < num_2D_partitions; i++) {
    colidx_partitioned[i].resize(nes_of_partitions[i]); // allocate memory for the edges in each partition
    for (size_t w = 0; w < degrees[i].size(); w++)
      rowptr_partitioned[i][w+1] = rowptr_partitioned[i][w] + degrees[i][w];
  }

  // insert edges to each partition
  std::vector<int> index(num_2D_partitions, 0);
  for (int v = 0; v < nv; v++) {
    auto src_cid = cluster_ids[v]; // src cluster id
    for (auto u : g->N(v)) {
      auto dst_cid = cluster_ids[u]; // dst cluster id
      int pid = src_cid * num_vertex_chunks + dst_cid; // partition id
      colidx_partitioned[pid][index[pid]++] = u;
    }
  }

  // compute the offsets for each partition
  std::vector<int> vlengths(num_2D_partitions, 0);
  std::vector<int> elengths(num_2D_partitions, 0);
  std::vector<int> voffsets(num_2D_partitions+1, 0);
  std::vector<int> eoffsets(num_2D_partitions+1, 0);
  for (int i = 0; i < num_2D_partitions; i ++) {
    vlengths[i] = rowptr_partitioned[i].size();
    elengths[i] = colidx_partitioned[i].size();
    //std::cout << "partition " << i << " has " << vlengths[i]-1 << " source vertices and " << elengths[i] << " edges\n";
  }
  parallel_prefix_sum<vidType,vidType>(vlengths, voffsets.data());
  parallel_prefix_sum<vidType,vidType>(elengths, eoffsets.data());

  // write rowptr_partitioned and colidx_partitioned into file, together with their offsets
  partitioned_file_path = g->get_inputfile_path() + '/';
  std::cout << "writing to path: " << partitioned_file_path << "\n";
  ofstream p_meta(partitioned_file_path+"pgraph.meta.txt");
  assert(p_meta);
  p_meta << voffsets[num_2D_partitions] << "\n" << eoffsets[num_2D_partitions] << "\n";
  p_meta.close();
  ofstream p_voffsets(partitioned_file_path+"pgraph.voffsets.bin", ios::out | ios::binary);
  ofstream p_eoffsets(partitioned_file_path+"pgraph.eoffsets.bin", ios::out | ios::binary);
  ofstream p_vertices(partitioned_file_path+"pgraph.vertex.bin", ios::out | ios::binary);
  ofstream p_edges(partitioned_file_path+"pgraph.edge.bin", ios::out | ios::binary);
  for (int i = 0; i < num_2D_partitions; i++) {
    size_t num = rowptr_partitioned[i].size();
    p_vertices.write((char *) &rowptr_partitioned[i][0], sizeof(int)*num);
    num = colidx_partitioned[i].size();
    p_edges.write((char *) &colidx_partitioned[i][0], sizeof(int)*num);
  }
  p_voffsets.write((char *) &voffsets[0], sizeof(int)*(num_2D_partitions+1));
  p_eoffsets.write((char *) &eoffsets[0], sizeof(int)*(num_2D_partitions+1));
}

// given the ids of clusters, fetch the edges between vertices in these clusters, and form a subgraph in CSR format
void PartitionedGraph::fetch_partitions(std::string path, std::vector<int> clusters, Graph *& subg) {
  int rowptr_size, colidx_size; // partitioned CSR size
  ifstream p_meta(path+"/pgraph.meta.txt");
  if (p_meta.fail())
    std::cerr << "Cannot find partitioned graph in " << path << ". Has this graph been partitioned yet?\n";
  assert(p_meta);
  p_meta >> rowptr_size >> colidx_size;
  ifstream p_vertices(path+"/pgraph.vertex.bin", ios::out | ios::binary);
  assert(p_vertices);
  ifstream p_edges(path+"/pgraph.edge.bin", ios::out | ios::binary);
  assert(p_edges);

  int *rowptr, *colidx, *voffsets, *eoffsets;
  map_file(path+"/pgraph.vertex.bin", rowptr, rowptr_size);
  map_file(path+"/pgraph.edge.bin", colidx, colidx_size);
  map_file(path+"/pgraph.voffsets.bin", voffsets, num_2D_partitions+1);
  map_file(path+"/pgraph.eoffsets.bin", eoffsets, num_2D_partitions+1);
  /*
     for (int i = 0; i < num_2D_partitions+1; i ++) {
     std::cout << "voffsets " << i << " is " << voffsets[i] << " eoffsets " << i << " is " << eoffsets[i] << "\n";
     }
     for (int i = 0; i < rowptr_size; i ++) {
     std::cout << "rowptr[" << i << "] = " << rowptr[i] << " ";
     }
     std::cout << "\n";
     for (int i = 0; i < colidx_size; i ++) {
     std::cout << "colidx[" << i << "] = " << colidx[i] << " ";
     }
     std::cout << "\n";
     */
  // count the number of vertices |V| in the subgraph
  int nv_subg = 0, ne_subg = 0;
  for (auto cid : clusters) nv_subg += verts_of_clusters[cid].size();
  std::cout << "number of vertices in the subgraph: " << nv_subg << "\n";

  int nc = clusters.size();
  int np = nc * nc;
  std::vector<std::vector<int>> rowptr_partitioned(np); // row pointers of each partition
  std::vector<std::vector<int>> colidx_partitioned(np); // column indices of each partition 
  std::vector<int> degrees_subg(nv_subg);
  int local_pid = 0;
  int vid_offset = 0;
  // compute |E| and the degrees for each vertex in the subgraph
  for (auto src_cid : clusters) {
    auto num_v = verts_of_clusters[src_cid].size();
    for (auto dst_cid : clusters) {
      int pid = src_cid * num_vertex_chunks + dst_cid; // partition id
      auto start = voffsets[pid];
      auto end = voffsets[pid+1];
      rowptr_partitioned[local_pid].insert(rowptr_partitioned[local_pid].end(), rowptr+start, rowptr+end);
      ne_subg += rowptr[end-1]; // the number of edges in this partition
      for (size_t i = 0; i < num_v; i++)
        degrees_subg[vid_offset+i] += rowptr_partitioned[local_pid][i+1] - rowptr_partitioned[local_pid][i];

      start = eoffsets[pid];
      end = eoffsets[pid+1];
      colidx_partitioned[local_pid].insert(colidx_partitioned[local_pid].end(), colidx+start, colidx+end);
      local_pid ++;
    }
    vid_offset += num_v;
  }
  std::cout << "number of edges in the subgraph: " << ne_subg << "\n";

  // gather global vertex ids in the given clusters
  VertexList vertices_subg;
  for (auto cid : clusters)
    vertices_subg.insert(vertices_subg.end(), verts_of_clusters[cid].begin(), verts_of_clusters[cid].end());
  assert(vertices_subg.size() == size_t(nv_subg));
  std::map<int, int> global_to_local_idmap;
  int local_id = 0;
  for (auto v : vertices_subg) {
    global_to_local_idmap.insert(std::make_pair(v, local_id++));
  }
  assert(local_id == nv_subg);
  //std::cout << "global to local vertex id map constructed\n";

  subg = new Graph(nv_subg, ne_subg); // allocate memory for the subgraph
  eidType *offsets = custom_alloc_global<eidType>(nv_subg+1);
  parallel_prefix_sum<vidType,eidType>(degrees_subg, offsets);

  // insert the edges into the subgraph
  int pid = 0;
  int src = 0; // src is local id of the source vertex
  for (auto src_cid : clusters) {
    for (auto v : verts_of_clusters[src_cid]) { // v is the global vertex id
      subg->fixEndEdge(src, offsets[src+1]); // fix row pointers
      auto idx = offsets[src];
      auto rank = vertex_rank_in_cluster[v];
      // for each dst cluster, copy the corresponding edges
      for (size_t pid_index = 0; pid_index < clusters.size(); pid_index++) {
        auto start = rowptr_partitioned[pid+pid_index][rank]; // the location of the first edge
        auto end = rowptr_partitioned[pid+pid_index][rank+1]; // number of edges to copy
        for (int i = start; i < end; i++) {
          auto global_dst = colidx_partitioned[pid+pid_index][i]; // This 'dst' is a global vertex ID. 
          auto dst = global_to_local_idmap[global_dst]; // translate into its local vertex ID
          if (dst >= nv_subg) std::cout << "global_dst = " << global_dst << ", dst = " << dst << ", nv_subg = " << nv_subg << "\n";
          assert(dst < nv_subg);
          subg->constructEdge(idx++, dst); // fix column indices
        }
      }
      src++;
    }
    pid += clusters.size();
  }
  std::cout << "Subgraph constructed\n";
  subg->print_graph();
  std::cout << "global to local vertex id mapping: ";
  for (auto pair : global_to_local_idmap)
    std::cout << "<" << pair.first << ", " << pair.second << "> ";
  std::cout << "\n";
}

