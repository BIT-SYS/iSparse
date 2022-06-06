#ifndef __KERNEL_H1__
#define __KERNEL_1__
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include "nvToolsExt.h"
#define FULL_MASK 0xffffffff
#define THREADS_PER_BLOCK 256
#define PERCENT 0.01

__global__ void count_GPU(
    uint num_rows,
    uint num_rows_per_block,
    int exp,
    uint *rowPtr,
    double *vals,
    uint *count);
__global__ void split_GPU(
    uint num_rows,
    uint num_rows_per_block,
    int exp,
    uint *rowPtr,
    uint *colInd,
    double *vals,
    uint *count,
    uint *rowPtrS,
    uint *colIndS,
    int32_t *valS,
    uint *rowPtrD,
    uint *colIndD,
    double *valD);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <unsigned int THREADS_PER_VECTOR>
__global__ void spmv_GPU_S(
    const unsigned int num_rows,
    const int exponent,
    const int32_t *csrValS,
    const double *xs,
    const uint *csrColIndS,
    const uint *csrRowPtrS,
    double *y)
{ //异步传输
    //每个block处理的vector数量，即一行
    const uint VECS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_VECTOR;

    const uint thread_id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x; // vector处理矩阵的一行
    const uint thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);     // vector中的线程号，相当于%31
    const uint vector_id = thread_id / THREADS_PER_VECTOR;               //该线程处理哪一行
    const uint vector_lane = threadIdx.x / THREADS_PER_VECTOR;           //处理block中的哪个vector
    const uint num_vectors = VECS_PER_BLOCK * gridDim.x;                 //每次循环所有block处理的vector数

    uint ptrsS0;
    uint ptrsS1;
    for (uint row = vector_id; row < num_rows; row += num_vectors)
    {

        if (thread_lane == 0)
        {
            ptrsS0 = csrRowPtrS[row + thread_lane];
            ptrsS1 = csrRowPtrS[row + thread_lane + 1];
        }

        const uint row_startS = __shfl_sync(0xffffffff, ptrsS0, vector_lane * THREADS_PER_VECTOR);
        const uint row_endS = __shfl_sync(0xffffffff, ptrsS1, vector_lane * THREADS_PER_VECTOR);

        double sum = 0;

        // single
        for (uint jj = row_startS + thread_lane; jj < row_endS; jj += THREADS_PER_VECTOR)
        {
            // int32_t halfval[2];
            // int32_t sig = csrValS[jj] & 0x80000000;
            // int32_t exp = (exponent + 1023) << 20;
            // int32_t tail = csrValS[jj] & 0x7fffffff;
            // int32_t tail1 = tail >> 11;
            // int32_t tail2 = tail & 0x7ff;
            // halfval[1] = sig | exp | tail1;
            // halfval[0] = tail2 << 21;
            // double *new_val = (double *)halfval;
            int32_t val = csrValS[jj];
            int32_t halfval[2];
            halfval[0] = (val & 0x7ff) << 21;
            int32_t sig = val & 0x80000000;
            int32_t exp = (exponent + 1023) << 20;
            halfval[1] = sig | exp | ((val & 0x7ffff800) >> 11);
            double *new_val = (double *)halfval;
            sum += (*new_val) * xs[csrColIndS[jj]];
        }

        for (int offset = THREADS_PER_VECTOR / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
        if (thread_lane == 0)
        {
            y[row] = sum;
        }
    }
}

template <unsigned int THREADS_PER_VECTOR>
__global__ void spmv_GPU_D(
    const unsigned int num_rows,
    const double *xd,
    double *y,
    const double *csrValD,
    const uint *csrColIndD,
    const uint *csrRowPtrD)
{ //异步传输
    //每个block处理的vector数量，即一行
    const uint VECS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_VECTOR;

    const uint thread_id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x; // vector处理矩阵的一行
    const uint thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);     // vector中的线程号，相当于%31
    const uint vector_id = thread_id / THREADS_PER_VECTOR;               //该线程处理哪一行
    const uint vector_lane = threadIdx.x / THREADS_PER_VECTOR;           //处理block中的哪个vector
    const uint num_vectors = VECS_PER_BLOCK * gridDim.x;                 //每次循环所有block处理的vector数

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

        for (uint jj = row_startD + thread_lane; jj < row_endD; jj += THREADS_PER_VECTOR)
        {
            sum += csrValD[jj] * xd[csrColIndD[jj]];
        }
        for (int offset = THREADS_PER_VECTOR / 2; offset > 0; offset /= 2)
        {
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
        }
        if (thread_lane == 0)
        {
            y[row] = sum;
        }
    }
}

//一个kernel内算 S D两种精度子矩阵
template <unsigned int THREADS_PER_VECTOR>
__global__ void spmv_GPU_SD(
    const unsigned int num_rows,
    const int exponent,
    const double *xd,
    double *y,
    const int32_t *csrValS,
    const uint *csrColIndS,
    const uint *csrRowPtrS,
    const double *csrValD,
    const uint *csrColIndD,
    const uint *csrRowPtrD)
{ //异步传输
    //每个block处理的vector数量，即一行
    const uint VECS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
    const uint thread_id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x; // vector处理矩阵的一行
    const uint thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);     // vector中的线程号，相当于%31
    const uint vector_id = thread_id / THREADS_PER_VECTOR;               //该线程处理哪一行
    const uint vector_lane = threadIdx.x / THREADS_PER_VECTOR;           //处理block中的哪个vector
    const uint num_vectors = VECS_PER_BLOCK * gridDim.x;                 //每次循环所有block处理的vector数
    uint ptrsS0;
    uint ptrsS1;
    uint ptrsD0;
    uint ptrsD1;
    for (uint row = vector_id; row < num_rows; row += num_vectors)
    {
        if (thread_lane == 0)
        {
            ptrsD0 = csrRowPtrD[row + thread_lane];
            ptrsD1 = csrRowPtrD[row + thread_lane + 1];
            ptrsS0 = csrRowPtrS[row + thread_lane];
            ptrsS1 = csrRowPtrS[row + thread_lane + 1];
        }

        const uint row_startD = __shfl_sync(0xffffffff, ptrsD0, vector_lane * THREADS_PER_VECTOR);
        const uint row_endD = __shfl_sync(0xffffffff, ptrsD1, vector_lane * THREADS_PER_VECTOR);
        const uint row_startS = __shfl_sync(0xffffffff, ptrsS0, vector_lane * THREADS_PER_VECTOR);
        const uint row_endS = __shfl_sync(0xffffffff, ptrsS1, vector_lane * THREADS_PER_VECTOR);
        double sum = 0;

        // single
        for (uint jj = row_startS + thread_lane; jj < row_endS; jj += THREADS_PER_VECTOR)
        {
            int32_t val = csrValS[jj];
            int32_t halfval[2];
            halfval[0] = (val & 0x7ff) << 21;
            int32_t sig = val & 0x80000000;
            int32_t exp = (exponent + 1023) << 20;
            halfval[1] = sig | exp | ((val & 0x7ffff800) >> 11);
            double *new_val = (double *)halfval;
            sum += (*new_val) * xd[csrColIndS[jj]];
        }
        for (uint jj = row_startD + thread_lane; jj < row_endD; jj += THREADS_PER_VECTOR)
        {
            sum += csrValD[jj] * xd[csrColIndD[jj]];
             
        }
        for (int offset = THREADS_PER_VECTOR / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
        if (thread_lane == 0)
        {
            y[row] = sum;
        }
    }
}

#endif