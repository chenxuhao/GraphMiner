import os
import re
import statistics

graph_name = "livej"

files = ["tc_sample_omp_base", "tc_sample_mask", "tc_sample_edge_mask", "tc_sample_color_sparse", "tc_sample_edge_sparse", "tc_sample_edge_stream_v0"]
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


for f in files[1:]:
    print('./bin/'+f+' inputs/' + graph_name + '/graph')
    times = []
    errors = []
    counts = []
    for i in range(5):
        var = os.popen('./bin/'+f+' inputs/' + graph_name + '/graph').read()
        m = re.search('total_num_pattern = ([0-9]+)', var)
        count = m.group(1)
        m = re.search('runtime \[\w+\] = ([0-9.]+) sec', var)
        time = m.group(1)
        errors.append(abs(float(count) - base) / base)
        counts.append(float(count))
        times.append(float(time))
    #print(times)
    #print(errors)
    print("mean time: " +  str(statistics.mean(times)))
    print("var time: "+ str(statistics.variance(times)))
    print("mean error: "+ str(statistics.mean(errors)))
    print("var error: "+ str(statistics.variance(errors)))
    print("var count: "+ str(statistics.variance(counts)))


