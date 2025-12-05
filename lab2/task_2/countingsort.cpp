#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NPROC
#define NPROC 10
#endif

//A cache line is the unit of data transfer between the cache and main memory (needed for solving the task). Typically the cache line is 64 bytes.
#ifndef LEVEL1_DCACHE_LINESIZE
#define LEVEL1_DCACHE_LINESIZE 64 //bytes
#endif

//formula as a macro to calculate padding
#define INTS_PER_CACHE_LINE (LEVEL1_DCACHE_LINESIZE / sizeof(int))
#define PADDED_ELEMENTS(k) (((k) + INTS_PER_CACHE_LINE - 1) / INTS_PER_CACHE_LINE * INTS_PER_CACHE_LINE)

template <const int k> void seq_countingsort(int *out, int const *in, const int n) {
    int counters[k] = {}; //all zeros
    for (int i = 0; i < n; ++i)
        ++counters[in[i]];
    int tmp, sum = 0;
    for (int i = 0; i < k; ++i) {
        tmp = counters[i];
        counters[i] = sum;
        sum += tmp;
    }
    for (int i = 0; i < n; ++i)
        out[counters[in[i]]++] = in[i];
}


template <const int k> void par_countingsort(int *out, int const *in, const int n) {
    int counters[NPROC][k] = {}; //all zeros
    #pragma omp parallel num_threads(NPROC)
    {   
        int *thcounters = counters[omp_get_thread_num()];
        #pragma omp for
        for (int i = 0; i < n; ++i)
            ++thcounters[in[i]];
        #pragma omp single
        {
            int tmp, sum = 0;
            for (int j = 0; j < k; ++j)
                for (int i = 0; i < NPROC; ++i) {
                    tmp = counters[i][j];
                    counters[i][j] = sum;
                    sum += tmp;
                }
        }
        #pragma omp for
        for (int i = 0; i < n; ++i)
            out[thcounters[in[i]]++] = in[i];
    }
}


template <const int k> void par_countingsort_padded(int *out, int const *in, const int n) {
    const int padded_k = PADDED_ELEMENTS(k);
    int* counters = (int*)aligned_alloc(LEVEL1_DCACHE_LINESIZE, NPROC * padded_k * sizeof(int));
    
    #pragma omp parallel for
    for (int i = 0; i < NPROC * padded_k; ++i) counters[i] = 0;
    
    #pragma omp parallel num_threads(NPROC)
    {   
        int tid = omp_get_thread_num();
        int *thcounters = counters + tid * padded_k;
        
        #pragma omp for
        for (int i = 0; i < n; ++i)
            ++thcounters[in[i]];
        
        #pragma omp barrier
        #pragma omp single
        {
            int sum = 0;
            for (int j = 0; j < k; ++j)
                for (int i = 0; i < NPROC; ++i) {
                    int* tcounters = counters + i * padded_k;
                    int tmp = tcounters[j];
                    tcounters[j] = sum;
                    sum += tmp;
                }
        }
        
        #pragma omp for
        for (int i = 0; i < n; ++i)
            out[thcounters[in[i]]++] = in[i];
    }
    
    free(counters);
}

#define CEILDIV(x,y) (((x)+(y)-1)/(y)) //return ceil(x/y)

bool checkreset(int *out, int const *in, const int n) {
    int insum = 0, outsum = 0, notsorted = 0;
    #pragma omp parallel for reduction(+:insum)
    for (int i = 0; i < n; ++i) insum += in[i];
    #pragma omp parallel for reduction(+:outsum)
    for (int i = 0; i < n; ++i) outsum += out[i];
    #pragma omp parallel for reduction(+:notsorted)
    for (int i = 1; i < n; ++i) notsorted += out[i-1]>out[i];
    if(insum!=outsum || notsorted) return false;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) out[i] = 0;
    return true;
}

#ifndef K
#define K 10
#endif

int main(int argc, char *argv[]) {
    
    printf("NPROC = %d\n", NPROC);
    printf("LEVEL1_DCACHE_LINESIZE = %d byte\n", LEVEL1_DCACHE_LINESIZE);
    printf("K = %d\n\n", K);
    
    //display padding calculation
    printf("INTS_PER_CACHE_LINE = %ld\n", (long)INTS_PER_CACHE_LINE);
    printf("PADDED_ELEMENTS(%d) = %ld\n\n", K, (long)PADDED_ELEMENTS(K));
    
    //init input
    const int n = atoi(argv[1]);
    int* in = (int*)malloc(sizeof(int)*n);
    int* out = (int*)aligned_alloc(LEVEL1_DCACHE_LINESIZE, sizeof(int)*n);
    for (int i = 0; i < n; ++i)
        in[i] = rand()%K;
    printf("n = %d\n", n);
    
    //tests
    double ts = omp_get_wtime();
    seq_countingsort<K>(out, in, n);
    ts = omp_get_wtime() - ts;
    printf("seq, elapsed time = %.3f seconds, check passed = %c\n", ts, checkreset(out,in,n)?'y':'n');
    
    double tp = omp_get_wtime();
    par_countingsort<K>(out, in, n);
    tp = omp_get_wtime() - tp;
    printf("par, elapsed time = %.3f seconds (%.1fx speedup), check passed = %c\n", tp, ts/tp, checkreset(out,in,n)?'y':'n');
    
    //test padded version
    double tpp = omp_get_wtime();
    par_countingsort_padded<K>(out, in, n);
    tpp = omp_get_wtime() - tpp;
    printf("par_padded, elapsed time = %.3f seconds (%.1fx speedup), check passed = %c\n", tpp, ts/tpp, checkreset(out,in,n)?'y':'n');
    printf("Improvement over naive parallel: %.1f%%\n", (tp/tpp - 1.0) * 100.0);
    
    //free mem
    free(in);
    free(out);
    
    return EXIT_SUCCESS;
}