import os
import re
import statistics

graph_name = "mico"

files = ["motif_omp_base", "motif_color_sparse_args"]

arg_vals = ["10", "5", "4","3","2", "1"]
depth = 1

def gen_args(depth):
    if(depth == 0):
        return [[]]
    
    output = []
    for i in gen_args(depth-1):
        for j in arg_vals:
            new_item = i[:]
            new_item.append(j)
            output.append(new_item)
    
    return output

#files = ["4cc_sample_omp_base", "4cc_sample_color_sparse", "4cc_sample_edge_sparse", "4cc_sample_edge_stream_v0"]
#files = ["clique_sample_omp_base", "clique_sample_color_sparse", "clique_sample_edge_sparse"]


print('./bin/'+files[0]+' inputs/' + graph_name + '/graph')
var = os.popen('./bin/'+files[0]+' inputs/' + graph_name + '/graph').read()
m = re.findall('pattern ([0-9]): ([0-9]+)\n', var)
base = [float(n) for i,n in m]
m = re.search('runtime \[\w+\] = ([0-9.]+)', var)
time = m.group(1)
base_time = float(time)

# base = 467429836174
# base_time = 145.491
print("base count: "+ str(base)+ "\nbase time: "+ str(base_time))

print(gen_args(depth))

results = {}
f = files[1]
for i in gen_args(depth):
    print('./bin/'+f+' inputs/' + graph_name + '/graph' + " 1 1024 " + " ".join(i))
    var = os.popen('./bin/'+f+' inputs/' + graph_name + '/graph' + " 1 1024 " + " ".join(i)).read()
    print(var)
    m = re.findall('pattern ([0-9]): ([0-9]+)\n', var)
    counts = [float(n) for l,n in m]
    m = re.search('runtime \[\w+\] = ([0-9.]+)', var)
    time = m.group(1)
    results[" ".join(i)] = [float(time), [abs(bi - float(ci)) / bi for bi,ci in zip(base,counts)]]

print(results)


output = []
for k in results:
    output.append({"args": k, "error": statistics.mean(results[k][1]), "time": results[k][0]})

output = sorted(output, key=lambda d: float(d['args'])) 

import json
print(json.dumps(output))




