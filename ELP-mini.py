  
import re
import statistics
import os

f = "./bin/clique6_sample_color_sparse_args"
graph = "inputs/mico/graph"

base_c = 50

## process -- run with preliminary c value on smaller graph n times
## estimate standard error of true count with sqrt(var/n)
## use 3*standard error over expected mean to get error on subgraph
## scale error on subgraph to estimate error on true count
## scale c based on error within graph to find c for error bound

while True:
    n=3

    if(base_c == 1):
        break

    times = []
    counts = []
    for i in range(n):
        #print(f + " " + graph + " 1 1024 1 " + str(int(base_c)))
        var = os.popen(f + " " + graph + " 1 1024 1 " + str(int(base_c))).read()
        m = re.search('total_num_pattern = ([0-9]+)', var)
        count = m.group(1)
        m = re.search('runtime \[\w+\] = ([0-9.]+) sec', var)
        time = m.group(1)
        times.append(float(time))
        counts.append(float(count))

    print("CNT", base_c)
    print("var",statistics.variance(counts)/n)
    print("mean",statistics.mean(counts))

    error = (max(counts) - min(counts)) / statistics.mean(counts)
    print(error)

    if(error < 0.1):
        break
    
    base_c -= error*10

print("C:", int(base_c), "Bounded Error:", error)


## todo: run on small graph
## scale variance
## do same error estimation
## scale down error by fraction of graph
