

void Graph::color_sparsify(vector<int> color_layers) {
  auto new_edges = new vidType[n_edges];
  auto colors = new int[color_layers.size()][n_vertices];

  for(int i = 0; i < color_layers.size(); i++) {

        for (vidType v = 0; v < n_vertices; v++) {
            colors[i][v] = rand() % c;
            //printf("color[%d] = %d, %d\n", v, colors[v], rand());
        }

        eidType count = 0;
        eidType edges_removed = 0;
        eidType last_offset = 0;
        for (vidType v = 0; v < n_vertices; v++) {
            auto begin = edge_begin(v); 
            auto end = edge_end(v);

            for(auto e = begin;  e < end; e++) {
            if(colors[v] != colors[edges[e]]) { //remove edge
                edges_removed += 1;
            } else {
                new_edges[count] = edges[e];
                count += 1;
            }
            }
            vertices[v] -= last_offset; // take out from end of last interval considering prev-removed edges
            last_offset = edges_removed;   
        }
        vertices[n_vertices] -= edges_removed;
        n_edges = count;
        edges = new_edges;

        }
  
}