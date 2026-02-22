Last name of Student 1: Khan
First name of Student 1:Zayan
Email of Student 1:zayan@ucsb.edu
Last name of Student 2:
First name of Student 2:
Email of Student 2:


If CSIL is used for performance assessment instead of Expanse, make sure you evaluate when such a machine is lightly 
loaded using “uptime”. Please  indicate your evaluation is done on CSIL and list the uptime index of that CSIL machine.  

Report 
----------------------------------------------------------------------------
1. How is the code parallelized? Show your solution by listing the key computation parallelized with
  OpenMP and related code. 
Parallelized the two per-iteration loops in parallel_itmv_mult():

#pragma omp parallel
for (k = 0; k < no_iterations; k++) {
  #pragma omp for schedule(runtime)
  for (i = 0; i < matrix_dim; i++) mv_compute(i);      // y = d + A x (row-wise)

  #pragma omp for schedule(runtime)
  for (i = 0; i < matrix_dim; i++) vector_x[i] = vector_y[i];   // x = y
}

Mapping method is implemented via omp_set_schedule(...) + schedule(runtime):
- BLOCK_MAPPING  -> static, chunk ≈ n/threads
- BLOCK_CYCLIC   -> static, chunk = 1 or 16

Upper-triangular optimization: mv_compute(i) starts inner loop at j=i, so it skips j<i
(lower-triangular zeros are never multiplied).

void parallel_itmv_mult(int threadcnt, int mappingtype, int chunksize) {
  int i, k, chunk = chunksize; omp_sched_t sched = omp_sched_static;
  if (mappingtype == BLOCK_MAPPING) { chunk = matrix_dim / threadcnt; if (chunk < 1) chunk = 1; sched = omp_sched_static; }
  else if (mappingtype == BLOCK_CYCLIC) { if (chunk < 1) chunk = 1; sched = omp_sched_static; }
  else if (mappingtype == BLOCK_DYNAMIC) { if (chunk < 1) chunk = 1; sched = omp_sched_dynamic; }

  omp_set_num_threads(threadcnt);
  omp_set_schedule(sched, chunk);

  #pragma omp parallel private(i,k)
  for (k = 0; k < no_iterations; k++) {
    #pragma omp for schedule(runtime)
    for (i = 0; i < matrix_dim; i++) mv_compute(i);
    #pragma omp for schedule(runtime)
    for (i = 0; i < matrix_dim; i++) vector_x[i] = vector_y[i];
  }
}

----------------------------------------------------------------------------
2.  Report the parallel time, speedup, and efficiency with blocking mapping, block cyclic mapping with block size 1 
and block size 16 using  2 cores (2 threads), and 4 cores (4 threads) for parallelizing the code 
in handling a full dense matrix with n=4096 and t=1024. 

Mapping                     Threads  Time(s)     GFLOPS   Speedup   Eff
BLOCK_MAPPING (9)              2     12.652995   2.7155   1.4757   0.7378
BLOCK_CYCLIC r=1 (10)          2     14.128432   2.4320   1.3063   0.6532
BLOCK_CYCLIC r=16 (11)         2     12.330008   2.7867   1.5622   0.7811

BLOCK_MAPPING (9)              4      4.982305   6.8964   3.7477   0.9369
BLOCK_CYCLIC r=1 (10)          4      5.987561   5.7385   3.0824   0.7706
BLOCK_CYCLIC r=16 (11)         4      5.947187   5.7775   3.2389   0.8097


----------------------------------------------------------------------------
3.  Report the parallel time, speedup, and efficiency with blocking mapping, block cyclic mapping with block size 1 
and block size 16 using  2 cores (2 threads), and 4 cores (4 threads) for parallelizing the code 
in handling an upper triangular matrix (n=4096 and t=1024).

Write a short explanation on why one mapping method is significantly faster than or similar to another.


Mapping                      Threads  Time(s)     GFLOPS   Speedup   Eff
BLOCK_MAPPING (12)             2      7.807578   2.2009   1.4114   0.7057
BLOCK_CYCLIC r=1 (13)          2      8.557685   2.0080   1.2041   0.6020
BLOCK_CYCLIC r=16 (14)         2      7.315016   2.3491   1.5154   0.7577

BLOCK_MAPPING (12)             4      4.172983   4.1179   2.6408   0.6602
BLOCK_CYCLIC r=1 (13)          4      3.594314   4.7809   2.8667   0.7167
BLOCK_CYCLIC r=16 (14)         4      2.902229   5.9210   3.8194   0.9549

Why mapping differs
- Dense: work per row is uniform, so BLOCK_MAPPING benefits from contiguous rows.
- Upper triangular: row i does ~ (n-i) work, so contiguous row blocks can be imbalanced. BLOCK_CYCLIC spreads heavy early rows across threads.
  Here, r=16 is fastest at 4 threads, giving the best balance. 