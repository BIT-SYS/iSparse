#ifndef __SPMV_LJ__
#define __SPMV_LJ__
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "sparseMatrix.h"
#include <cuda_runtime.h>
#define THREADS_PER_BLOCK_SPMV 512
#define FULL_MASK 0xffffffff

template <uint THREADS_PER_VECTOR>
__global__ void spmv_GPU_D(
    const uint num_rows,
    const double *xd,
    double *y,
    const double *csrValD,
    const uint *csrColIndD,
    const uint *csrRowPtrD)
{ //异步传输
    //每个block处理的vector数量，即一行
    const uint VECS_PER_BLOCK = THREADS_PER_BLOCK_SPMV / THREADS_PER_VECTOR;
    __shared__ volatile double sdata[VECS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];

    const uint thread_id = THREADS_PER_BLOCK_SPMV * blockIdx.x + threadIdx.x; // vector处理矩阵的一行
    const uint thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // vector中的线程号，相当于%31
    const uint vector_id = thread_id / THREADS_PER_VECTOR;                    //该线程处理哪一行
    const uint vector_lane = threadIdx.x / THREADS_PER_VECTOR;                //处理block中的哪个vector
    const uint num_vectors = VECS_PER_BLOCK * gridDim.x;                      //每次循环所有block处理的vector数

    uint ptrsD0;
    uint ptrsD1;

    for (uint row = vector_id; row < num_rows; row += num_vectors)
    {
        if (thread_lane == 0)
        {
            ptrsD0 = csrRowPtrD[row + thread_lane];
            ptrsD1 = csrRowPtrD[row + thread_lane + 1];
        }

        const uint row_startD = __shfl_sync(0xffffffff, ptrsD0, vector_lane * THREADS_PER_VECTOR);
        const uint row_endD = __shfl_sync(0xffffffff, ptrsD1, vector_lane * THREADS_PER_VECTOR);

        double sum = 0;

        // double
        if (THREADS_PER_VECTOR == 32 && row_endD - row_startD > 32)
        {
            uint jj = row_startD - (row_startD & (THREADS_PER_VECTOR - 1)) + thread_lane;
            if (jj >= row_startD && jj < row_endD)
                sum += csrValD[jj] * xd[csrColIndD[jj]];
            for (jj += THREADS_PER_VECTOR; jj < row_endD; jj += THREADS_PER_VECTOR)
            {
                sum += csrValD[jj] * xd[csrColIndD[jj]];
            }
        }
        else
        {
            for (uint jj = row_startD + thread_lane; jj < row_endD; jj += THREADS_PER_VECTOR)
            {
                sum += csrValD[jj] * xd[csrColIndD[jj]];
            }
        }

        for (int offset = THREADS_PER_VECTOR / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(FULL_MASK, sum, offset);

        if (thread_lane == 0)
        {
            y[row] = sum;
        }
    }
}

void spmv(SpM<double> *A, double *x, double *y);
#endif