import os
import re
import statistics

graph_name = "livej"

files = ["clique4_omp_base", "tc_sample_perf_args"]

arg_vals = [0.1, 0.3, 0.5, 0.9]
#files = ["4cc_sample_omp_base", "4cc_sample_color_sparse", "4cc_sample_edge_sparse", "4cc_sample_edge_stream_v0"]
#files = ["clique_sample_omp_base", "clique_sample_color_sparse", "clique_sample_edge_sparse"]

print('./bin/'+files[0]+' inputs/' + graph_name + '/graph')
var = os.popen('./bin/'+files[0]+' inputs/' + graph_name + '/graph').read()
m = re.search('total_num_pattern = ([0-9]+)', var)
count = m.group(1)
m = re.search('runtime \[\w+\] = ([0-9.]+) sec', var)
time = m.group(1)
base = float(count)
base_time = float(time)

# base = 467429836174
# base_time = 145.491
print("base count: "+ str(base)+ "\nbase time: "+ str(base_time))


results = {}
f = files[1]
for i in arg_vals:
    for j in arg_vals:
            print('./bin/'+f+' inputs/' + graph_name + '/graph' + " 1 1024 1 " + str(i) + " " + str(j))
            var = os.popen('./bin/'+f+' inputs/' + graph_name + '/graph' + " 1 1024 1 " + str(i) + " " + str(j)).read()
            m = re.search('total_num_pattern = ([0-9]+)', var)
            count = m.group(1)
            m = re.search('runtime \[\w+\] = ([0-9.]+) sec', var)
            time = m.group(1)
            results[str(i) + " " + str(j)] = [float(time), abs(base - float(count)) / base]

print(results)


output = []
for k in results:
    output.append({"args": k, "error": results[k][1], "time": results[k][0]})

output = sorted(output, key=lambda d: d['args']) 

import json
print(json.dumps(output))



