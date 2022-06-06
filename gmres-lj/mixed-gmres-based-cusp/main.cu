#include <cusp/csr_matrix.h>
#include <cusp/monitor.h>
#include <cusp/krylov/gmres.h>
#include <cusp/gallery/poisson.h>
#include "cusp/io/matrix_market.h"
#include <stdio.h>
#include <sparseMatrix.h>
extern "C"
{
#include "mmio.h"
}
int main(uint argc, char *argv[])
{
    // argv[1] matrix path
    // create an empty sparse matrix structure (CSR format)
    SpM A1;
    A1.readMtx(argv[1]);
    // printf("num_entries: %d\n", A1.nrows);
    assert(A1.nrows == A1.ncols);
    // printf("%s  %d  %d  %d\n",argv[1],A1.nrows,A1.ncols,A1.nnz);
    cusp::csr_matrix<uint, double, cusp::host_memory> A;
    A.num_cols = A1.ncols;
    A.num_rows = A1.nrows;
    A.num_entries = A1.nnz;
    for (int i = 0; i < A.num_rows + 1; i++)
    {
        A.row_offsets.push_back(A1.rows[i]);
    }
    for (int i = 0; i < A.num_entries; i++)
    {
        A.column_indices.push_back(A1.cols[i]);
    }
    for (int i = 0; i < A.num_entries; i++)
    {
        A.values.push_back(A1.vals[i]);
    }
    free(A1.rows);
    free(A1.cols);
    free(A1.vals);
    // readMtx(argv[1], A);
    // initialize matrix
    // cusp::gallery::poisson5pt(A, 10, 10);

    // cusp::io::read_matrix_market_file(A, argv[1]);
    // printf("num_entries: %d\n", A.num_rows);
    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<double, cusp::host_memory> x(A.num_rows, 0);
    cusp::array1d<double, cusp::host_memory> b(A.num_rows, 1);

    // set stopping criteria:
    //  iteration_limit    = 100
    //  relative_tolerance = 1e-6
    //  absolute_tolerance = 0
    //  verbose            = true
    cusp::monitor<double> monitor(b, 2000, 1e-6, 0, true);
    uint restart = 20;
    // set preconditioner (identity)
    cusp::identity_operator<double, cusp::host_memory> M(A.num_rows, A.num_rows);
    // printf("num_entries: %d\n", M.num_entries);
    // solve the linear system A x = b

    // cusp::csr_matrix<unsigned int,double,cusp::device_memory>B(A);
    cudaEvent_t start_event, stop_event;
    float cuda_elapsed_timed;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);
    cusp::krylov::gmres(A, x, b, restart, monitor, M);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&cuda_elapsed_timed, start_event, stop_event);

    FILE *fp;
    fp = fopen("/home/liujie/gmres-lj/mixed-gmres-based-cusp/gmres-cusp-out1", "a+");
    fprintf(fp, "%s %f ms\n", argv[1], cuda_elapsed_timed);
    fclose(fp);
    return 0;
}
