extern "C"
{
#include "mmio.h"
}
#include "sparseMatrix.h"
#include <cuda_runtime.h>
#include <cusp/monitor.h>
#include <iostream>
// #include <cusp/krylov/gmres.h>
#include "cusp/io/matrix_market.h"
#ifndef VALUETYPE
#define VALUETYPE double
#endif
using namespace std;
extern double norm2(uint n, double *a);
extern void gmres_mixed(SpM<double> *A, double *x, double *b, uint restart); //传入的都是cpu上的数据;
int main(int argc, char *argv[])
{
    SpM<double> A;
    // argv[1] matrix path
    A.readMtx(argv[1]);
    printf("M=%u\tN=%u\tNNZ=%u\n", A.nrows, A.ncols, A.nnz);
    double *x;
    x = (double *)malloc(sizeof(double) * A.nrows);
    double *b;
    b = (double *)malloc(sizeof(double) * A.nrows);

    // cusp::array1d<double, cusp::device_memory> x1(A.nrows);
    // cusp::array1d<double, cusp::device_memory> b1(A.ncols);

    srand(time(NULL));
    for (uint i = 0; i < A.nrows; i++)
    {
        // b[i] = (1 * rand() / (RAND_MAX + 1.0));
        b[i] = 1.0;
        x[i] = 0.0;
        // cout<<A.vals[i]<<' '<<A.vals[i*2+1]<<endl;
        // b1[i] = b[i];
    }
    cudaEvent_t start_event, stop_event;
    float cuda_elapsed_timed;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);
    gmres_mixed(&A, x, b, 20);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&cuda_elapsed_timed, start_event, stop_event);

    printf("%f ms\n", cuda_elapsed_timed);
    free(A.rows);
    free(A.cols);
    free(A.vals);
    free(x);
    free(b);

    // cusp::monitor<double> monitor(b1, 100, 1e-6, 0, true);
    // cusp::csr_matrix<int, double, cusp::device_memory> A1;
    // cusp::io::read_matrix_market_file(A1, argv[1]);
    // cudaEventRecord(start_event, 0);
    // cusp::krylov::gmres(A1, x1, b1, 20, monitor);
    // cudaEventRecord(stop_event, 0);
    // cudaEventSynchronize(stop_event);
    // cudaEventElapsedTime(&cuda_elapsed_timed, start_event, stop_event);
    // printf("%f ms\n", cuda_elapsed_timed);
    return 0;
}