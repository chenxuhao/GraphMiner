# GraphGPUMiner
Graph Pattern Mining (GPM) Framework on GPU.

Three existing state-of-the-art frameworks:

Pangolin [1]: source code is in src/pangolin/.

SgMatch [2,3]: https://github.com/guowentian/SubgraphMatchGPU

Peregrine [4]: https://github.com/pdclab/peregrine

[1] Xuhao Chen, Roshan Dathathri, Gurbinder Gill, Keshav Pingali,
Pangolin: An Efficient and Flexible Graph Pattern Mining System on CPU and GPU. VLDB 2020

[2] GPU-Accelerated Subgraph Enumeration on Partitioned Graphs. SIGMOD 2020.

[3] Exploiting Reuse for GPU Subgraph Enumeration. Under submission. TKDE 2020.

[4] Kasra Jamshidi, Rakesh Mahadasa, Keval Vora,
Peregrine: A Pattern-Aware Graph Mining System. EuroSys 2020


Quick Start
-----------

Install CUDA 11.1.1 and GCC 8.3.1. 
If CUDA version < 11.0, enable CUB in the Makefile.
Note: the latest official CUB requires CUDA 11+. For CUDA version < 11.0, use CUB v1.8.0.

Setup CUB library:

    $ git submodule update --init --recursive

Go to each sub-directory, e.g. src/triangle, and then

    $ cd src/triangle; make

Find out commandline format by running executable without argument:

    $ cd ../../bin
    $ ./tc_omp_base

Run triangle counting on an undirected graph:

    $ ./tc_omp_base ../inputs/citeseer/graph

To control the number of threads, set the following environment variable:

    $ export OMP_NUM_THREADS=[ number of cores in system ]


Graph Loading
-------------

The graph loading infrastructure understands the following formats:

+ `graph.meta.txt` text file specifying the number of vertices, edges and maximum degree

+ `graph.vertex.bin` binary file containing the row pointers

+ `graph.edge.bin` binary file containing the column indices


Citation
-------------

Please cite the following paper if you use this code:

```
@article{Pangolin,
	title={Pangolin: An Efficient and Flexible Graph Mining System on CPU and GPU},
	author={Xuhao Chen and Roshan Dathathri and Gurbinder Gill and Keshav Pingali},
	year={2020},
	journal = {Proc. VLDB Endow.},
	issue_date = {August 2020},
	volume = {13},
	number = {8},
	month = aug,
	year = {2020},
	numpages = {12},
	publisher = {VLDB Endowment},
}
```

