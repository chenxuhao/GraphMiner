import os
import re
import statistics

graph_name = "livej"

files = ["house_sample_color_sparse_args","clique5_sample_color_sparse_args", "clique6_sample_color_sparse_args"]
base = [13892452066046, 1995295290765, 7872561225874]

index = 0


arg_vals = ["15"]
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




# base = 467429836174
# base_time = 145.491

print(gen_args(depth))

results = {}
f = files[index]
for i in gen_args(depth):
    times = []
    errors = []
    for j in range(5):
        print('./bin/'+f+' /mnt/md0/hb_inputs/' + graph_name + '/graph' + " 1 1024 1 " + " ".join(i))
        var = os.popen('./bin/'+f+' /mnt/md0/hb_inputs/' + graph_name + '/graph' + " 1 1024 1 " + " ".join(i)).read()
        m = re.search('total_num_pattern = ([0-9]+)', var)
        count = m.group(1)
        m = re.search('runtime \[\w+\] = ([0-9.]+) sec', var)
        time = m.group(1)
        times.append(float(time))
        errors.append(abs(base[index] - float(count)) / base[index])
    results[" ".join(i)] = [times, errors]



output = []
for k in results:
    output.append({"args": k, "error": statistics.mean(results[k][1]), "time": statistics.mean(results[k][0])})


output = sorted(output, key=lambda d: float(d['args'])) 

import json
print(json.dumps(output))




