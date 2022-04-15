# How to reproduce our OSDI'2022 paper results

To begin with, download the [datasets](https://www.dropbox.com/sh/i1jq1uwtkcd2qo0/AADJck_u3kx7FeSR5BvdrkqYa?dl=0) and clone this repository.
The first 3 graphs (`Mico`, `Patent_citations`, `Youtube`) are vertex-labeled graphs which are used for FSM.
Put the datasets in the `inputs` directory.

On GPU, we compare with [Pangolin](src/pangolin/) and [PBE](https://github.com/guowentian/SubgraphMatchGPU).
We also compare with CPU-targeted systems [Peregrine](https://github.com/pdclab/peregrine) and `GraphZero` (the OpenMP version in this repository).

Next, we will build G<sup>2</sup>Miner. This requires [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) 11.1.1 or greater.
It also requires GCC 8 or greater. To install GCC 9 on Ubuntu 18.04, run:

```
$ sudo add-apt-repository ppa:ubuntu-toolchain-r/test
$ sudo apt update
$ sudo apt install gcc-9 g++-9
$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
```

Now we are ready to compile G<sup>2</sup>Miner. Simply make in the root directory:

```
$ make
```

Binaries will be in the `bin` directory. 
For example, `tc_omp_base` is the OpenMP version of triangle counting on CPU, `tc_gpu_base` is the single GPU version, and `tc_multigpu` is the multi-GPU version.
Finally, we can move on to the experiments.

### Main benchmarks

All these experiments were conducted on the NVIDIA V100 GPU. For CPU, we use Intel Xeon Gold 5120 2.2GHz CPUs (56 cores in total) and 190GB RAM,

Note that the paper reports the mean runtimes of three executions for each experiment; here we only show each command once.

The total execution time can be found at the end of the log file.

We evaluate five workloads, triangle counting (TC), k-clique listing (k-CL), subgraph listing (SL), k-motif counting (k-MC), and frequent subgraph mining (FSM). 
For SL, we use two patterns `rectangle` (i.e., `4-cycle`) and `diamond`.

#### Triangle Counting (Table 4)

```
$ bin/tc_gpu_base inputs/livej/graph      > lj-tc.log 2>&1
$ bin/tc_gpu_base inputs/orkut/graph      > ok-tc.log 2>&1
$ bin/tc_gpu_base inputs/twitter20/graph  > tw2-tc.log 2>&1
$ bin/tc_gpu_base inputs/twitter40/graph  > tw4-tc.log 2>&1
$ bin/tc_gpu_base inputs/friendster/graph > fr-tc.log 2>&1
$ bin/tc_gpu_base inputs/uk2007/graph     > uk-tc.log 2>&1
```

#### Clique Listing (Table 5)

```
$ # 4-cliques
$ bin/clique_gpu_base inputs/livej/graph      4 > lj-4-cliques.log 2>&1
$ bin/clique_gpu_base inputs/orkut/graph      4 > ok-4-cliques.log 2>&1
$ bin/clique_gpu_base inputs/twitter20/graph  4 > tw2-4-cliques.log 2>&1
$ bin/clique_gpu_base inputs/twitter40/graph  4 > tw4-4-cliques.log 2>&1
$ bin/clique_gpu_base inputs/friendster/graph 4 > fr-4-cliques.log 2>&1
$ # 5-cliques
$ bin/clique_gpu_base inputs/livej/graph      5 > lj-5-cliques.log 2>&1
$ bin/clique_gpu_base inputs/orkut/graph      5 > ok-5-cliques.log 2>&1
$ bin/clique_gpu_base inputs/friendster/graph 5 > fr-5-cliques.log 2>&1
```

#### Subgraph Listing (Table 6)

```
$ # diamond
$ bin/sg_gpu_base inputs/livej/graph diamond        > lj-diamond.log 2>&1
$ bin/sg_gpu_base inputs/orkut/graph diamond        > or-diamond.log 2>&1
$ bin/sg_gpu_base inputs/twitter20/graph diamond    > tw2-diamond.log 2>&1
$ bin/sg_gpu_base inputs/twitter40/graph diamond    > tw4-diamond.log 2>&1
$ bin/sg_gpu_base inputs/friendster/graph diamond   > fr-diamond.log 2>&1
$ # rectangle
$ bin/sg_gpu_base inputs/livej/graph rectangle      > lj-rectangle.log 2>&1
$ bin/sg_gpu_base inputs/orkut/graph rectangle      > or-rectangle.log 2>&1
$ bin/sg_gpu_base inputs/friendster/graph rectangle > fr-rectangle.log 2>&1
```

#### Motif Counting (Table 7)

```
$ # 3-motifs
$ bin/motif_gpu_base inputs/livej/graph      3 > lj-3-motifs.log 2>&1
$ bin/motif_gpu_base inputs/orkut/graph      3 > ok-3-motifs.log 2>&1
$ bin/motif_gpu_base inputs/twitter20/graph  3 > tw2-3-motifs.log 2>&1
$ bin/motif_gpu_base inputs/twitter40/graph  3 > tw4-3-motifs.log 2>&1
$ bin/motif_gpu_base inputs/friendster/graph 3 > fr-3-motifs.log 2>&1
$ # 4-motifs
$ bin/motif_gpu_base inputs/livej/graph      4 > lj-4-motifs.log 2>&1
$ bin/motif_gpu_base inputs/orkut/graph      4 > ok-4-motifs.log 2>&1
$ bin/motif_gpu_base inputs/friendster/graph 4 > fr-4-motifs.log 2>&1
```

#### Frequent Subgraph Mining (Table 8)

Notice that FSM takes two more arguments: `max_num_edges` an `minimum_support`. For 3-FSM, `max_num_edges` should be set to 2.

```
$ # Mico
$./bin/fsm_gpu_base inputs/mico/graph 2 300  > mi-fsm-3-300.log 2>&1
$./bin/fsm_gpu_base inputs/mico/graph 2 500  > mi-fsm-3-500.log 2>&1
$./bin/fsm_gpu_base inputs/mico/graph 2 1000 > mi-fsm-3-1000.log 2>&1
$./bin/fsm_gpu_base inputs/mico/graph 2 5000 > mi-fsm-3-5000.log 2>&1
$ # Patent
$./bin/fsm_gpu_base inputs/patent_citations/graph 2 300  > pa-fsm-3-300.log 2>&1
$./bin/fsm_gpu_base inputs/patent_citations/graph 2 500  > pa-fsm-3-500.log 2>&1
$./bin/fsm_gpu_base inputs/patent_citations/graph 2 1000 > pa-fsm-3-1000.log 2>&1
$./bin/fsm_gpu_base inputs/patent_citations/graph 2 5000 > pa-fsm-3-5000.log 2>&1
$ # Youtube
$./bin/fsm_gpu_base inputs/youtube/graph 2 300  > yo-fsm-3-300.log 2>&1
$./bin/fsm_gpu_base inputs/youtube/graph 2 500  > yo-fsm-3-500.log 2>&1
$./bin/fsm_gpu_base inputs/youtube/graph 2 1000 > yo-fsm-3-1000.log 2>&1
$./bin/fsm_gpu_base inputs/youtube/graph 2 5000 > yo-fsm-3-5000.log 2>&1
```
