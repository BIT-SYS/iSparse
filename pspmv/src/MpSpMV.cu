/*
 *  Copyright NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "format.h"
#include <cusp/system/cuda/arch.h>
#define FULL_MASK 0xffffffff
#define min(a, b) ((a) < (b) ? (a) : (b))

#define THREADS_PER_BLOCK 128

extern int NUM_ITERATIONS;

template <unsigned int THREADS_PER_VECTOR>
__global__ void spmv_GPU_SD_Mp(const unsigned int num_rows,
                               const float *AxS,
                               const float *xS,
                               const double *xD,
                               const uint *AjS,
                               const uint *ApS,
                               double *y,
                               const double *AxD,
                               const uint *AjD,
                               const uint *ApD)
{

    const size_t VECTORS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_VECTOR;

    const uint thread_id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x; // global thread index
    const uint thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);     // thread index within the vector
    const uint vector_id = thread_id / THREADS_PER_VECTOR;               // global vector index
    const uint vector_lane = threadIdx.x / THREADS_PER_VECTOR;           // vector index within the block
    const uint num_vectors = VECTORS_PER_BLOCK * gridDim.x;              // total number of active vectors
    uint ptrsS0;
    uint ptrsS1;
    uint ptrsD0;
    uint ptrsD1;
    for (uint row = vector_id; row < num_rows; row += num_vectors)
    {
        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version

        if (thread_lane == 0)
        {
            ptrsD0 = ApD[row + thread_lane];
            ptrsD1 = ApD[row + thread_lane + 1];
            ptrsS0 = ApS[row + thread_lane];
            ptrsS1 = ApS[row + thread_lane + 1];
        }

        const uint row_startD = __shfl_sync(0xffffffff, ptrsD0, vector_lane * THREADS_PER_VECTOR);
        const uint row_endD = __shfl_sync(0xffffffff, ptrsD1, vector_lane * THREADS_PER_VECTOR);
        const uint row_startS = __shfl_sync(0xffffffff, ptrsS0, vector_lane * THREADS_PER_VECTOR);
        const uint row_endS = __shfl_sync(0xffffffff, ptrsS1, vector_lane * THREADS_PER_VECTOR);
        // initialize local sum
        double sum = 0.0;
        // accumulate local sums

        // Single precision

        // accumulate local sums
        for (uint jj = row_startS + thread_lane; jj < row_endS; jj += THREADS_PER_VECTOR)
        {
            sum += AxS[jj] * xS[AjS[jj]];
        }

        // Double precision
        //  accumulate local sums
        for (uint jj = row_startD + thread_lane; jj < row_endD; jj += THREADS_PER_VECTOR)
        {
            sum += AxD[jj] * xD[AjD[jj]];
        }

        for (int offset = THREADS_PER_VECTOR / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
        if (thread_lane == 0)
        {
            y[row] = sum;
        }
    }
}

void Mp_SpMV(SpM *A, float *xs, double *xd, double *y_Mp, double *split_time_Mp, double *transfer_time_Mp, double *SpMV_time_Mp, uint *nnzs, uint *nnzd, int *vec)
{
    SpMS AS;
    SpM AD;

    cudaEvent_t start_event, stop_event;
    float cuda_elapsed_time;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    size_t coo1_nnz = 0;
    size_t coo2_nnz = 0;

    for (uint i = 0; i < A->nnz; i++)
    {
        if (A->vals[i] > -1 && A->vals[i] < 1)
        {
            coo2_nnz++;
        }
        else
        {
            coo1_nnz++; //双精度
        }
    }
    (*nnzs) = coo2_nnz;
    (*nnzd) = coo1_nnz;

    AD.ncols = A->ncols;
    AD.nrows = A->nrows;
    AD.nnz = coo1_nnz;
    AD.rows = (uint *)malloc(sizeof(uint) * (1 + AD.nrows));
    AD.cols = (uint *)malloc(sizeof(uint) * coo1_nnz);
    AD.vals = (double *)malloc(sizeof(double) * coo1_nnz);

    AS.ncols = A->ncols;
    AS.nrows = A->nrows;
    AS.nnz = coo2_nnz;
    AS.rows = (uint *)malloc(sizeof(uint) * (1 + AS.nrows));
    AS.cols = (uint *)malloc(sizeof(uint) * coo2_nnz);
    AS.vals = (float *)malloc(sizeof(float) * coo2_nnz);

    coo1_nnz = 0;
    coo2_nnz = 0;

    // timing split loop

    AS.rows[0] = 0;
    AD.rows[0] = 0;
    cudaEventRecord(start_event, 0); //开始计时——预处理时间
    for (uint i = 0; i < A->nrows; i++)
    {
        for (uint j = A->rows[i]; j < A->rows[i + 1]; j++)
        {
            if (A->vals[j] > -1 && A->vals[j] < 1)
            {
                AS.cols[coo2_nnz] = A->cols[j];
                AS.vals[coo2_nnz] = A->vals[j];
                coo2_nnz++;
            }
            else
            {
                AD.cols[coo1_nnz] = A->cols[j];
                AD.vals[coo1_nnz] = A->vals[j];
                coo1_nnz++;
            }
        }
        AS.rows[i + 1] = coo2_nnz;
        AD.rows[i + 1] = coo1_nnz;
    }
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&cuda_elapsed_time, start_event, stop_event);
    (*split_time_Mp) = cuda_elapsed_time;

    uint i;

    uint *d_rowPtrS;
    uint *d_colIndS;
    float *d_vals;

    uint *d_rowPtrD;
    uint *d_colIndD;
    double *d_vald;

    float *d_xs;
    double *d_xd;

    double *d_y;

    cudaMalloc(((void **)(&d_vals)), AS.nnz * sizeof(float));
    cudaMalloc(((void **)(&d_colIndS)), AS.nnz * sizeof(uint));
    cudaMalloc(((void **)(&d_rowPtrS)), (A->nrows + 1) * sizeof(uint));

    cudaMalloc(((void **)(&d_xs)), A->ncols * sizeof(float));
    cudaMalloc(((void **)(&d_xd)), A->ncols * sizeof(double));
    cudaMalloc(((void **)(&d_y)), A->nrows * sizeof(double));

    cudaMalloc(((void **)(&d_vald)), AD.nnz * sizeof(double));
    cudaMalloc(((void **)(&d_colIndD)), AD.nnz * sizeof(uint));
    cudaMalloc(((void **)(&d_rowPtrD)), (A->nrows + 1) * sizeof(uint));

    int max = 32;

    (*vec) = max;

    size_t MAX_BLOCKS = 0;
    if (max == 2)
    {
        MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_SD_Mp<2>, THREADS_PER_BLOCK, 0);
    }
    else if (max == 4)
    {
        MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_SD_Mp<4>, THREADS_PER_BLOCK, 0);
    }
    else if (max == 8)
    {
        MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_SD_Mp<8>, THREADS_PER_BLOCK, 0);
    }
    else if (max == 16)
    {
        MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_SD_Mp<16>, THREADS_PER_BLOCK, 0);
    }
    else
    {
        MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_SD_Mp<32>, THREADS_PER_BLOCK, 0);
    }

    const size_t VECTORS_PER_BLOCK = THREADS_PER_BLOCK / max;
    uint min_num = min(MAX_BLOCKS, (A->nrows + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
    const size_t NUM_BLOCKS = min_num < 1 ? 1 : min_num;

    cudaEventRecord(start_event, 0); //开始计时——传输时间
    for (i = 0; i < NUM_ITERATIONS; i++)
    {
        cudaMemcpy(d_xs, xs, A->ncols * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_xd, xd, A->ncols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vals, AS.vals, AS.nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIndS, AS.cols, AS.nnz * sizeof(uint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rowPtrS, AS.rows, (A->nrows + 1) * sizeof(uint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vald, AD.vals, AD.nnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIndD, AD.cols, AD.nnz * sizeof(uint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rowPtrD, AD.rows, (A->nrows + 1) * sizeof(uint), cudaMemcpyHostToDevice);
    }
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&cuda_elapsed_time, start_event, stop_event);
    (*transfer_time_Mp) = cuda_elapsed_time / NUM_ITERATIONS;

    /*************计算threads_per_vector*************************************************************************************/
    cudaEventRecord(start_event, 0); //开始计时——计算时间
    for (i = 0; i < NUM_ITERATIONS; i++)
    {
        if (max == 2)
        {
            spmv_GPU_SD_Mp<2><<<NUM_BLOCKS, THREADS_PER_BLOCK, 0>>>(A->nrows, d_vals, d_xs, d_xd, d_colIndS, d_rowPtrS, d_y, d_vald, d_colIndD, d_rowPtrD);
        }
        else if (max == 4)
        {
            spmv_GPU_SD_Mp<4><<<NUM_BLOCKS, THREADS_PER_BLOCK, 0>>>(A->nrows, d_vals, d_xs, d_xd, d_colIndS, d_rowPtrS, d_y, d_vald, d_colIndD, d_rowPtrD);
        }
        else if (max == 8)
        {
            spmv_GPU_SD_Mp<8><<<NUM_BLOCKS, THREADS_PER_BLOCK, 0>>>(A->nrows, d_vals, d_xs, d_xd, d_colIndS, d_rowPtrS, d_y, d_vald, d_colIndD, d_rowPtrD);
        }
        else if (max == 16)
        {
            spmv_GPU_SD_Mp<16><<<NUM_BLOCKS, THREADS_PER_BLOCK, 0>>>(A->nrows, d_vals, d_xs, d_xd, d_colIndS, d_rowPtrS, d_y, d_vald, d_colIndD, d_rowPtrD);
        }
        else
        {
            spmv_GPU_SD_Mp<32><<<NUM_BLOCKS, THREADS_PER_BLOCK, 0>>>(A->nrows, d_vals, d_xs, d_xd, d_colIndS, d_rowPtrS, d_y, d_vald, d_colIndD, d_rowPtrD);
        }
    }
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&cuda_elapsed_time, start_event, stop_event);
    (*SpMV_time_Mp) = cuda_elapsed_time;

    cudaMemcpy(y_Mp, d_y, A->nrows * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_vals);
    cudaFree(d_xs);
    cudaFree(d_xd);
    cudaFree(d_rowPtrS);
    cudaFree(d_y);
    cudaFree(d_vald);
    cudaFree(d_colIndD);
    cudaFree(d_rowPtrD);

    free(A->rows);
    free(AS.rows);
    free(AD.rows);
    free(A->cols);
    free(AS.cols);
    free(AD.cols);
    free(A->vals);
    free(AS.vals);
    free(AD.vals);
}
