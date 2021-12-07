extern "C"
{
#include "mmio.h"
}
#include "sparseMatrix.h"
#include <cuda_runtime.h>
#include <cusp/monitor.h>
#ifndef VALUETYPE
#define VALUETYPE double
#endif
extern double norm2(uint n, double *a);
extern void gmres(SpM<double> *A, double *x, double *b, uint restart); //传入的都是cpu上的数据;
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
    for (uint i = 0; i < A.nrows; i++)
    {
        x[i] = 1;
        b[i] = 1;
    }
    gmres(&A, x, b, 20);
    free(A.rows);
    free(A.cols);
    free(A.vals);
    free(x);
    free(b);
    return 0;
}