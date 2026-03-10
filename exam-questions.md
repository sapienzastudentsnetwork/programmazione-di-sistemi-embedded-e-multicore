# Oral Exam

## Part 1: Project Presentation

The oral exam starts with the presentation of the project:

* Be concise when overviewing the parallelization strategy: go in depth later if asked.
* Discuss the profiling you've done.
* Discuss the results obtained: go over the tables/plots in your report.

> If you make optimizations not discussed in the course (e.g., Vectorization, CUDA Streams, ...), expect to be asked about them.

## Part 2: Theory Questions/Exercises

If you did the project in a group, expect to receive questions on the arguments your colleague talked about in the report (e.g., if you talked about MPI+OMP you will probably receive questions on CUDA).

### Generic

* What is false sharing?
* Difference between Array of Structs (AOS) and Struct of Arrays (SOA).
* Between AOS and SOA, which layout occupies more space in memory?
* Given a percentage of sequential code, what is the maximum speedup achievable?
* Amdahl and Gustafson laws.
* What is the Roofline Model?
* What is Tiling?
* What is Pinned Memory?
* How is a read-write lock managed?

### MPI

* `MPI_Status` fields: in which cases do we need to get the missing tag or rank using `MPI_Status`?
* MPI Errors.
* How do we get the number of elements we have received in a communication? Why could we receive fewer elements than `buf_size`?
* What are the different levels of threading in MPI? (funneled, serialized, ...). What's the difference?
* What are derived datatypes in MPI?
* Discuss the different types of `MPI_Send`.
* How would you sum up two matrices using MPI?
* When could you have a deadlock in MPI? Which problem does `MPI_SendRecv` solve?
* How does `MPI_IN_PLACE` work?
* How does communication between GPUs work in MPI?

### OpenMP

* How do nested loops work and how do we manage them in OpenMP?
* How does variable scoping work in OpenMP?
* How does scheduling work in OpenMP?
* Parallelize a loop which performs the element-wise product of two arrays with OpenMP.
* Parallelize the following `for` cycles:

```C
for(int i = 2; i < N; i++){
 A[i] = A[i - 2] + A[i] * 0.5; 
}

//------------------------------

for(int i = 1; i < N; i++){
  for(int j = 1; j < M; j++){
     d[i][j] = d[i][j-1] + d[i-1][j-1] + d[i-1][j];
  }
}
```

### CUDA

* What are memory banks? When do we have bank conflicts and bank broadcast?
* When does it make sense to disable L1 cache in CUDA?
* Why does false sharing not happen in CUDA?
* Write a CUDA kernel which sums two arrays.
* Write a CUDA kernel which multiplies two arrays.
* What does it mean that memory accesses are coalesced?

### Pthread

* How does synchronization work in pthread?
