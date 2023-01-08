# Space-efficient-Computation-of-Parallel-Approximate-String-Matching


There are two versions of the program (1) OpenMP based (2) CUDA-Based

Sequences can be provided in fasta format as well as plain text format.

Maximum allowed memory can be adjusted using the constant MAXIMUM_MEM_ALLOWED but it cannot be less than size of the text.


## Compilig and Running an OpenMP program
g++ osm_mem.cpp -o osm -fopenmp -O3

to run: ./osm <sequence 1 file name>  <sequence 2 file name>  <threshold percentage>


## Compilig and Running a CUDA program
nvcc esm6.cu -Xcompiler -fopenmp -o esm -fopenmp -O3

to run: ./esm <sequence 1 file name>  <sequence 2 file name>  <threshold percentage>
  
## Please cite this article
  Space-efficient computation of parallel approximate string matching
  https://doi.org/10.1007/s11227-022-05038-6

## Please cite this article
  Space-efficient computation of parallel approximate string matching (The Journal of Supercomputing)
  
  https://doi.org/10.1007/s11227-022-05038-6

