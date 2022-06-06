/*
 *  Copyright 2011 The Regents of the University of California
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

/*! \file gmres.h
 *  \brief Generalized Minimum Residual (GMRES) method
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/detail/execution_policy.h>

#include <cstddef>
#include <sparseMatrix.h>
#include <cuda_fp16.h>
#include <cusp/system/cuda/arch.h>

namespace cusp
{
namespace krylov
{

/*! \addtogroup iterative_solvers Iterative Solvers
 *  \addtogroup krylov_methods Krylov Methods
 *  \ingroup iterative_solvers
 *  \{
 */

/* \cond */

template <typename DerivedPolicy,
          typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor,
          typename Preconditioner>
void gmres(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
           const LinearOperator& A,
                 VectorType1& x,
           const VectorType2& b,
           const size_t restart,
                 Monitor& monitor,
                 Preconditioner& M);

template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor>
void gmres(const LinearOperator& A,
                 VectorType1& x,
           const VectorType2& b,
           const size_t restart,
                 Monitor& monitor);

template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2>
void gmres(const LinearOperator& A,
                 VectorType1& x,
           const VectorType2& b,
           const size_t restart);

/* \endcond */

/**
 * \brief GMRES method
 *
 * \tparam LinearOperator is a matrix or subclass of \p linear_operator
 * \tparam VectorType1 vector
 * \tparam Monitor is a monitor such as \p default_monitor or \p verbose_monitor
 * \tparam Preconditioner is a matrix or subclass of \p linear_operator
 *
 * \param A matrix of the linear system
 * \param x approximate solution of the linear system
 * \param b right-hand side of the linear system
 * \param restart the method every restart inner iterations
 * \param monitor montiors iteration and determines stopping conditions
 * \param M preconditioner for A
 *
 * \par Overview
 * Solves the nonsymmetric, linear system A x = b
 * with preconditioner \p M.
 *
 * \par Example
 *
 *  The following code snippet demonstrates how to use \p gmres to
 *  solve a 10x10 Poisson problem.
 *
 *  \code
 *  #include <cusp/csr_matrix.h>
 *  #include <cusp/monitor.h>
 *  #include <cusp/krylov/gmres.h>
 *  #include <cusp/gallery/poisson.h>
 *
 *  int main(void)
 *  {
 *      // create an empty sparse matrix structure (CSR format)
 *      cusp::csr_matrix<int, float, cusp::device_memory> A;
 *
 *      // initialize matrix
 *      cusp::gallery::poisson5pt(A, 10, 10);
 *
 *      // allocate storage for solution (x) and right hand side (b)
 *      cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);
 *      cusp::array1d<float, cusp::device_memory> b(A.num_rows, 1);
 *
 *      // set stopping criteria:
 *      //  iteration_limit    = 100
 *      //  relative_tolerance = 1e-6
 *      //  absolute_tolerance = 0
 *      //  verbose            = true
 *      cusp::monitor<float> monitor(b, 100, 1e-6, 0, true);
 *      int restart = 50;
 *
 *      // set preconditioner (identity)
 *      cusp::identity_operator<float, cusp::device_memory> M(A.num_rows, A.num_rows);
 *
 *      // solve the linear system A x = b
 *      cusp::krylov::gmres(A, x, b,restart, monitor, M);
 *
 *      return 0;
 *  }
 *  \endcode

 *  \see \p monitor
 *
 */
template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor,
          typename Preconditioner>
void gmres(const LinearOperator& A,
                 VectorType1& x,
           const VectorType2& b,
           const size_t restart,
                 Monitor& monitor,
                 Preconditioner& M);
/*! \}
*/

//================================================================================================
#define FULL_MASK 0xffffffff
#define THREADS_PER_BLOCK_MY 128
template <unsigned int THREADS_PER_VECTOR>
__global__ void spmv_GPU_S(
    const unsigned int num_rows,
    const float *csrValS,
    const double *xs,
    const unsigned int *csrColIndS,
    const unsigned int *csrRowPtrS,
    double *y)
{ //异步传输
    //每个block处理的vector数量，即一行
    const int VECS_PER_BLOCK = THREADS_PER_BLOCK_MY / THREADS_PER_VECTOR;
    __shared__ volatile double sdata[VECS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];
    __shared__ volatile int ptrsS[VECS_PER_BLOCK][2];

    const int thread_id = THREADS_PER_BLOCK_MY * blockIdx.x + threadIdx.x; // vector处理矩阵的一行
    const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);     // vector中的线程号，相当于%31
    const int vector_id = thread_id / THREADS_PER_VECTOR;               //该线程处理哪一行
    const int vector_lane = threadIdx.x / THREADS_PER_VECTOR;           //处理block中的哪个vector
    const int num_vectors = VECS_PER_BLOCK * gridDim.x;                 //每次循环所有block处理的vector数
    float rdata = 0;
    double xdata = 0;
    float xfloat = 0;
    unsigned int ptrsS0;
    unsigned int ptrsS1;
    for (unsigned int row = vector_id; row < num_rows; row += num_vectors)
    {

        if (thread_lane == 0)
        {
            ptrsS0 = csrRowPtrS[row + thread_lane];
            ptrsS1 = csrRowPtrS[row + thread_lane + 1];
        }

        const unsigned int row_startS = __shfl_sync(0xffffffff, ptrsS0, vector_lane * THREADS_PER_VECTOR);
        const unsigned int row_endS = __shfl_sync(0xffffffff, ptrsS1, vector_lane * THREADS_PER_VECTOR);

        double sum = 0;

        for (unsigned int jj = row_startS + thread_lane; jj < row_endS; jj += THREADS_PER_VECTOR)
        {
            sum += (double)csrValS[jj] * xs[csrColIndS[jj]];
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
    const unsigned int *csrColIndD,
    const unsigned int *csrRowPtrD)
{ //异步传输
    //每个block处理的vector数量，即一行
    const int VECS_PER_BLOCK = THREADS_PER_BLOCK_MY / THREADS_PER_VECTOR;
    __shared__ volatile double sdata[VECS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];
    // __shared__ volatile int ptrsD[VECS_PER_BLOCK][2];

    const int thread_id = THREADS_PER_BLOCK_MY * blockIdx.x + threadIdx.x; // vector处理矩阵的一行
    const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);     // vector中的线程号，相当于%31
    const int vector_id = thread_id / THREADS_PER_VECTOR;               //该线程处理哪一行
    const int vector_lane = threadIdx.x / THREADS_PER_VECTOR;           //处理block中的哪个vector
    const int num_vectors = VECS_PER_BLOCK * gridDim.x;                 //每次循环所有block处理的vector数

    unsigned int ptrsD0;
    unsigned int ptrsD1;
    for (unsigned int row = vector_id; row < num_rows; row += num_vectors)
    {
        if (thread_lane == 0)
        {
            ptrsD0 = csrRowPtrD[row + thread_lane];
            ptrsD1 = csrRowPtrD[row + thread_lane + 1];
        }

        const unsigned int row_startD = __shfl_sync(0xffffffff, ptrsD0, vector_lane * THREADS_PER_VECTOR);
        const unsigned int row_endD = __shfl_sync(0xffffffff, ptrsD1, vector_lane * THREADS_PER_VECTOR);

        double sum = 0;

        // double
        if (THREADS_PER_VECTOR == 32 && row_endD - row_startD > 32)
        {
            unsigned int jj = row_startD - (row_startD & (THREADS_PER_VECTOR - 1)) + thread_lane;
            if (jj >= row_startD && jj < row_endD)
                sum += csrValD[jj] * xd[csrColIndD[jj]];
            for (jj += THREADS_PER_VECTOR; jj < row_endD; jj += THREADS_PER_VECTOR)
            {
                sum += csrValD[jj] * xd[csrColIndD[jj]];
            }
        }
        else
        {
            for (unsigned int jj = row_startD + thread_lane; jj < row_endD; jj += THREADS_PER_VECTOR)
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

template <unsigned int THREADS_PER_VECTOR>
__global__ void spmv_GPU_Hs(
    const unsigned int num_rows,
    const float *xh,
    double *y,
    const half *csrValH,
    const unsigned int *csrColIndH,
    const unsigned int *csrRowPtrH)
{ //异步传输
    //每个block处理的vector数量，即一行
    const int VECS_PER_BLOCK = THREADS_PER_BLOCK_MY / THREADS_PER_VECTOR;

    const int thread_id = THREADS_PER_BLOCK_MY * blockIdx.x + threadIdx.x; // vector处理矩阵的一行
    const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);     // vector中的线程号，相当于%31
    const int vector_id = thread_id / THREADS_PER_VECTOR;               //该线程处理哪一行
    const int vector_lane = threadIdx.x / THREADS_PER_VECTOR;           //处理block中的哪个vector
    const int num_vectors = VECS_PER_BLOCK * gridDim.x;                 //每次循环所有block处理的vector数

    double sum = 0;
    half sumh = 0;
    int var = 0;
    unsigned int ptrsH0;
    unsigned int ptrsH1;
    for (unsigned int row = vector_id; row < num_rows; row += num_vectors)
    {
        if (thread_lane == 0)
        {
            ptrsH0 = csrRowPtrH[row + thread_lane];
            ptrsH1 = csrRowPtrH[row + thread_lane + 1];
        }

        const unsigned int row_startH = __shfl_sync(0xffffffff, ptrsH0, vector_lane * THREADS_PER_VECTOR);
        const unsigned int row_endH = __shfl_sync(0xffffffff, ptrsH1, vector_lane * THREADS_PER_VECTOR);

        for (unsigned int jj = row_startH + thread_lane; jj < row_endH; jj += THREADS_PER_VECTOR)
        {
            sum += __half2float(csrValH[jj]) * xh[csrColIndH[jj]];
        }

        for (int offset = THREADS_PER_VECTOR / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
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
    const double *xd,
    double *y,
    const float *csrValS,
    const unsigned int *csrColIndS,
    const unsigned int *csrRowPtrS,
    const double *csrValD,
    const unsigned int *csrColIndD,
    const unsigned int *csrRowPtrD)
{ //异步传输
    //每个block处理的vector数量，即一行
    const int VECS_PER_BLOCK = THREADS_PER_BLOCK_MY / THREADS_PER_VECTOR;
    __shared__ volatile double sdata[VECS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];
    __shared__ volatile int ptrsS[VECS_PER_BLOCK][2];
    __shared__ volatile int ptrsD[VECS_PER_BLOCK][2];

    const int thread_id = THREADS_PER_BLOCK_MY * blockIdx.x + threadIdx.x; // vector处理矩阵的一行
    const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);     // vector中的线程号，相当于%31
    const int vector_id = thread_id / THREADS_PER_VECTOR;               //该线程处理哪一行
    const int vector_lane = threadIdx.x / THREADS_PER_VECTOR;           //处理block中的哪个vector
    const int num_vectors = VECS_PER_BLOCK * gridDim.x;                 //每次循环所有block处理的vector数
    unsigned int ptrsS0;
    unsigned int ptrsS1;
    unsigned int ptrsD0;
    unsigned int ptrsD1;
    for (unsigned int row = vector_id; row < num_rows; row += num_vectors)
    {
        if (thread_lane == 0)
        {
            ptrsD0 = csrRowPtrD[row + thread_lane];
            ptrsD1 = csrRowPtrD[row + thread_lane + 1];
            ptrsS0 = csrRowPtrS[row + thread_lane];
            ptrsS1 = csrRowPtrS[row + thread_lane + 1];
        }

        const unsigned int row_startD = __shfl_sync(0xffffffff, ptrsD0, vector_lane * THREADS_PER_VECTOR);
        const unsigned int row_endD = __shfl_sync(0xffffffff, ptrsD1, vector_lane * THREADS_PER_VECTOR);
        const unsigned int row_startS = __shfl_sync(0xffffffff, ptrsS0, vector_lane * THREADS_PER_VECTOR);
        const unsigned int row_endS = __shfl_sync(0xffffffff, ptrsS1, vector_lane * THREADS_PER_VECTOR);
        double sum = 0;

        // single
        for (unsigned int jj = row_startS + thread_lane; jj < row_endS; jj += THREADS_PER_VECTOR)
        {

            sum += csrValS[jj] * xd[csrColIndS[jj]];
        }

        // double
        for (unsigned int jj = row_startD + thread_lane; jj < row_endD; jj += THREADS_PER_VECTOR)
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

//一个kernel内算H S两种精度子矩阵
template <unsigned int THREADS_PER_VECTOR>
__global__ void spmv_GPU_HS(
    const unsigned int num_rows,
    const float *xs,
    const double *xd,
    double *y,
    const float *csrValS,
    const unsigned int *csrColIndS,
    const unsigned int *csrRowPtrS,
    const half *csrValH,
    const unsigned int *csrColIndH,
    const unsigned int *csrRowPtrH)
{ //异步传输
    //每个block处理的vector数量，即一行
    const int VECS_PER_BLOCK = THREADS_PER_BLOCK_MY / THREADS_PER_VECTOR;
    __shared__ volatile double sdata[VECS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];
    __shared__ volatile int ptrsS[VECS_PER_BLOCK][2];
    __shared__ volatile int ptrsH[VECS_PER_BLOCK][2];

    const int thread_id = THREADS_PER_BLOCK_MY * blockIdx.x + threadIdx.x; // vector处理矩阵的一行
    const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);     // vector中的线程号，相当于%31
    const int vector_id = thread_id / THREADS_PER_VECTOR;               //该线程处理哪一行
    const int vector_lane = threadIdx.x / THREADS_PER_VECTOR;           //处理block中的哪个vector
    const int num_vectors = VECS_PER_BLOCK * gridDim.x;                 //每次循环所有block处理的vector数
    unsigned int ptrsH0;
    unsigned int ptrsH1;
    unsigned int ptrsS0;
    unsigned int ptrsS1;
    for (unsigned int row = vector_id; row < num_rows; row += num_vectors)
    {
        if (thread_lane == 0)
        {
            ptrsH0 = csrRowPtrH[row + thread_lane];
            ptrsH1 = csrRowPtrH[row + thread_lane + 1];
            ptrsS0 = csrRowPtrS[row + thread_lane];
            ptrsS1 = csrRowPtrS[row + thread_lane + 1];
        }

        const unsigned int row_startH = __shfl_sync(0xffffffff, ptrsH0, vector_lane * THREADS_PER_VECTOR);
        const unsigned int row_endH = __shfl_sync(0xffffffff, ptrsH1, vector_lane * THREADS_PER_VECTOR);
        const unsigned int row_startS = __shfl_sync(0xffffffff, ptrsS0, vector_lane * THREADS_PER_VECTOR);
        const unsigned int row_endS = __shfl_sync(0xffffffff, ptrsS1, vector_lane * THREADS_PER_VECTOR);
        double sum = 0;
        for (unsigned int jj = row_startH + thread_lane; jj < row_endH; jj += THREADS_PER_VECTOR)
        {
            sum += __half2float(csrValH[jj]) * xs[csrColIndH[jj]];
        }

        for (unsigned int jj = row_startS + thread_lane; jj < row_endS; jj += THREADS_PER_VECTOR)
        {

            sum += csrValS[jj] * xd[csrColIndS[jj]];
        }

        for (int offset = THREADS_PER_VECTOR / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
        if (thread_lane == 0)
        {
            y[row] = sum;
        }
    }
}

//一个kernel内算H D两种精度子矩阵
template <unsigned int THREADS_PER_VECTOR>
__global__ void spmv_GPU_HD(
    const unsigned int num_rows,
    const float *xs,
    const double *xd,
    double *y,
    const double *csrValD,
    const unsigned int *csrColIndD,
    const unsigned int *csrRowPtrD,
    const half *csrValH,
    const unsigned int *csrColIndH,
    const unsigned int *csrRowPtrH)
{ //异步传输
    //每个block处理的vector数量，即一行
    const int VECS_PER_BLOCK = THREADS_PER_BLOCK_MY / THREADS_PER_VECTOR;
    __shared__ volatile double sdata[VECS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];
    __shared__ volatile int ptrsH[VECS_PER_BLOCK][2];
    __shared__ volatile int ptrsD[VECS_PER_BLOCK][2];

    const int thread_id = THREADS_PER_BLOCK_MY * blockIdx.x + threadIdx.x; // vector处理矩阵的一行
    const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);     // vector中的线程号，相当于%31
    const int vector_id = thread_id / THREADS_PER_VECTOR;               //该线程处理哪一行
    const int vector_lane = threadIdx.x / THREADS_PER_VECTOR;           //处理block中的哪个vector
    const int num_vectors = VECS_PER_BLOCK * gridDim.x;                 //每次循环所有block处理的vector数
    unsigned int ptrsH0;
    unsigned int ptrsH1;
    unsigned int ptrsD0;
    unsigned int ptrsD1;
    for (unsigned int row = vector_id; row < num_rows; row += num_vectors)
    {
        if (thread_lane == 0)
        {
            ptrsH0 = csrRowPtrH[row + thread_lane];
            ptrsH1 = csrRowPtrH[row + thread_lane + 1];
            ptrsD0 = csrRowPtrD[row + thread_lane];
            ptrsD1 = csrRowPtrD[row + thread_lane + 1];
        }

        const unsigned int row_startH = __shfl_sync(0xffffffff, ptrsH0, vector_lane * THREADS_PER_VECTOR);
        const unsigned int row_endH = __shfl_sync(0xffffffff, ptrsH1, vector_lane * THREADS_PER_VECTOR);
        const unsigned int row_startD = __shfl_sync(0xffffffff, ptrsD0, vector_lane * THREADS_PER_VECTOR);
        const unsigned int row_endD = __shfl_sync(0xffffffff, ptrsD1, vector_lane * THREADS_PER_VECTOR);
        double sum = 0;

        for (unsigned int jj = row_startH + thread_lane; jj < row_endH; jj += THREADS_PER_VECTOR)
        {
            sum += __half2float(csrValH[jj]) * xs[csrColIndH[jj]];
        }

        // double
        for (unsigned int jj = row_startD + thread_lane; jj < row_endD; jj += THREADS_PER_VECTOR)
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
//一个kernel内算三种精度子矩阵
template <unsigned int THREADS_PER_VECTOR>
__global__ void spmv_GPU_HSD(
    const unsigned int num_rows,
    const float *xs,
    const double *xd,
    double *y,
    const float *csrValS,
    const unsigned int *csrColIndS,
    const unsigned int *csrRowPtrS,
    const double *csrValD,
    const unsigned int *csrColIndD,
    const unsigned int *csrRowPtrD,
    const half *csrValH,
    const unsigned int *csrColIndH,
    const unsigned int *csrRowPtrH)
{ //异步传输
    //每个block处理的vector数量，即一行
    const int VECS_PER_BLOCK = THREADS_PER_BLOCK_MY / THREADS_PER_VECTOR;
    __shared__ volatile double sdata[VECS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];
    __shared__ volatile int ptrsH[VECS_PER_BLOCK][2];
    // __shared__ volatile int ptrsH[VECS_PER_BLOCK][6];
    __shared__ volatile int ptrsS[VECS_PER_BLOCK][2];
    __shared__ volatile int ptrsD[VECS_PER_BLOCK][2];

    const int thread_id = THREADS_PER_BLOCK_MY * blockIdx.x + threadIdx.x; // vector处理矩阵的一行
    const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);     // vector中的线程号，相当于%31
    const int vector_id = thread_id / THREADS_PER_VECTOR;               //该线程处理哪一行
    const int vector_lane = threadIdx.x / THREADS_PER_VECTOR;           //处理block中的哪个vector
    const int num_vectors = VECS_PER_BLOCK * gridDim.x;                 //每次循环所有block处理的vector数
    unsigned int ptrsH0;
    unsigned int ptrsH1;
    unsigned int ptrsS0;
    unsigned int ptrsS1;
    unsigned int ptrsD0;
    unsigned int ptrsD1;
    for (unsigned int row = vector_id; row < num_rows; row += num_vectors)
    {

        if (thread_lane == 0)
        {
            ptrsH0 = csrRowPtrH[row + thread_lane];
            ptrsH1 = csrRowPtrH[row + thread_lane + 1];
            ptrsS0 = csrRowPtrS[row + thread_lane];
            ptrsS1 = csrRowPtrS[row + thread_lane + 1];
            ptrsD0 = csrRowPtrD[row + thread_lane];
            ptrsD1 = csrRowPtrD[row + thread_lane + 1];
        }

        const unsigned int row_startH = __shfl_sync(0xffffffff, ptrsH0, vector_lane * THREADS_PER_VECTOR);
        const unsigned int row_endH = __shfl_sync(0xffffffff, ptrsH1, vector_lane * THREADS_PER_VECTOR);
        const unsigned int row_startS = __shfl_sync(0xffffffff, ptrsS0, vector_lane * THREADS_PER_VECTOR);
        const unsigned int row_endS = __shfl_sync(0xffffffff, ptrsS1, vector_lane * THREADS_PER_VECTOR);
        const unsigned int row_startD = __shfl_sync(0xffffffff, ptrsD0, vector_lane * THREADS_PER_VECTOR);
        const unsigned int row_endD = __shfl_sync(0xffffffff, ptrsD1, vector_lane * THREADS_PER_VECTOR);
        double sum = 0;

        for (unsigned int jj = row_startH + thread_lane; jj < row_endH; jj += THREADS_PER_VECTOR)
        {
            sum += __half2float(csrValH[jj]) * xs[csrColIndH[jj]];
        }

        for (unsigned int jj = row_startS + thread_lane; jj < row_endS; jj += THREADS_PER_VECTOR)
        {

            sum += csrValS[jj] * xd[csrColIndS[jj]];
        }

        for (unsigned int jj = row_startD + thread_lane; jj < row_endD; jj += THREADS_PER_VECTOR)
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

__global__ void count_GPU(
    unsigned int num_rows,
    unsigned int num_rows_per_block,
    unsigned int *rowPtr,
    double *vals,
    unsigned int *count)
{
    const int thread_id = THREADS_PER_BLOCK_MY * blockIdx.x + threadIdx.x;
    const int num_rows_per_thread = (num_rows_per_block + THREADS_PER_BLOCK_MY - 1) / THREADS_PER_BLOCK_MY;
    // const int block_edge = num_rows_per_block * (blockIdx.x + 1);
    unsigned int begin = num_rows_per_block * blockIdx.x + threadIdx.x * num_rows_per_thread;
    unsigned int end = num_rows_per_block * blockIdx.x + (threadIdx.x + 1) * num_rows_per_thread;
    if (end >= num_rows)
    {
        end = num_rows;
    }

    unsigned int count_h = 0;
    unsigned int count_s = 0;
    unsigned int count_d = 0;
    int exponent;
    int32_t *halfval;

    for (unsigned int i = begin; i < end; i++)
    {
        for (unsigned int j = rowPtr[i]; j < rowPtr[i + 1]; j++)
        {
            halfval = (int32_t *)(vals + j);
            exponent = ((halfval[1] >> 20) & 0x7ff) - 1023;
            // printf("%d\n",exponent);
            if (vals[j] == 0)
            {
                count_h++; //半精度
            }
            else if (exponent >= -15 && exponent <= 15)
            {
                // printf("%d  %d  %d\n",(halfval[1] & 0x3ff),halfval[0],(halfval[0] & 0x1fffffff));
                if ((halfval[1] & 0x3ff) == 0 && halfval[0] == 0)
                {
                    count_h++; //半精度
                }
                else if ((halfval[0] & 0x1fffffff) == 0)
                {
                    count_s++; //单精度
                }
                else
                {
                    count_d++; //双精度
                }
            }
            else if (exponent >= -127 && exponent <= 127)
            {
                if ((halfval[0] & 0x1fffffff) == 0)
                {
                    count_s++; //单精度
                }
                else
                {
                    count_d++; //双精度
                }
            }
            else
            {
                count_d++; //双精度
            }
        }
    }

    count[thread_id * 3] = count_h;
    count[thread_id * 3 + 1] = count_s;
    count[thread_id * 3 + 2] = count_d;
    // }
}

__global__ void split_GPU(
    unsigned int num_rows,
    unsigned int num_rows_per_block,
    unsigned int *rowPtr,
    unsigned int *colInd,
    double *vals,
    unsigned int *count,
    unsigned int *rowPtrH,
    unsigned int *colIndH,
    half *valH,
    unsigned int *rowPtrS,
    unsigned int *colIndS,
    float *valS,
    unsigned int *rowPtrD,
    unsigned int *colIndD,
    double *valD)
{
    const int thread_id = THREADS_PER_BLOCK_MY * blockIdx.x + threadIdx.x;
    const int num_rows_per_thread = (num_rows_per_block + THREADS_PER_BLOCK_MY - 1) / THREADS_PER_BLOCK_MY;
    // const int block_edge = num_rows_per_block * (blockIdx.x + 1);
    unsigned int begin = num_rows_per_block * blockIdx.x + threadIdx.x * num_rows_per_thread;
    unsigned int end = num_rows_per_block * blockIdx.x + (threadIdx.x + 1) * num_rows_per_thread;
    if (end >= num_rows)
    {
        end = num_rows;
    }
    // if (end >= block_edge)
    // {
    //     end = block_edge;
    // }

    unsigned int count_h = count[thread_id * 3];
    unsigned int count_s = count[thread_id * 3 + 1];
    unsigned int count_d = count[thread_id * 3 + 2];

    int exponent;
    int32_t *halfval;

    for (unsigned int i = begin; i < end; i++)
    {
        // if (thread_id == 1)
        // {
        //     printf("%d  %d  %d\n", count_h, count_s, count_d);
        // }
        rowPtrH[i] = count_h;
        rowPtrS[i] = count_s;
        rowPtrD[i] = count_d;
        for (unsigned int j = rowPtr[i]; j < rowPtr[i + 1]; j++)
        {
            halfval = (int32_t *)(vals + j);
            exponent = ((halfval[1] >> 20) & 0x7ff) - 1023;
            if (vals[j] == 0)
            {
                colIndH[count_h] = colInd[j];
                valH[count_h] = __float2half(vals[j]);
                count_h++; //半精度
            }
            else if (exponent >= -15 && exponent <= 15)
            {
                if ((halfval[1] & 0x3ff) == 0 && halfval[0] == 0)
                {
                    colIndH[count_h] = colInd[j];
                    valH[count_h] = __float2half(vals[j]);
                    count_h++; //半精度
                }
                else if ((halfval[0] & 0x1fffffff) == 0)
                {
                    colIndS[count_s] = colInd[j];
                    valS[count_s] = vals[j];
                    count_s++; //单精度
                }
                else
                {
                    colIndD[count_d] = colInd[j];
                    valD[count_d] = vals[j];
                    count_d++; //双精度
                }
            }
            else if (exponent >= -127 && exponent <= 127)
            {
                if ((halfval[0] & 0x1fffffff) == 0)
                {
                    colIndS[count_s] = colInd[j];
                    valS[count_s] = vals[j];
                    count_s++; //单精度
                }
                else
                {
                    colIndD[count_d] = colInd[j];
                    valD[count_d] = vals[j];
                    count_d++; //单精度
                }
            }
            else
            {
                colIndD[count_d] = colInd[j];
                valD[count_d] = vals[j];
                count_d++; //单精度
            }
        }
    }

    if (end == num_rows && begin < num_rows)
    {
        rowPtrH[num_rows] = count_h;
        rowPtrS[num_rows] = count_s;
        rowPtrD[num_rows] = count_d;
        // printf("%d  %d  %d\n", count_h, count_s, count_d);
    }
}


void mixed_split(const cusp::csr_matrix<unsigned int, double, cusp::host_memory> *A, SpMH *AH, SpMS *AS, SpM *AD)
{
    size_t csrh_nnz = 0;
    size_t csrs_nnz = 0;
    size_t csrd_nnz = 0;
    // const int THREADS_PER_BLOCK_MY = 512;

    int tmp1 = cusp::system::cuda::detail::max_active_blocks(split_GPU, THREADS_PER_BLOCK_MY, 0);
    int tmp2 =  cusp::system::cuda::detail::max_active_blocks(count_GPU, THREADS_PER_BLOCK_MY, 0);
    int max_blocks = tmp1 < tmp2 ? tmp1 : tmp2;

    int tmp3 = (A->num_rows + THREADS_PER_BLOCK_MY - 1) / THREADS_PER_BLOCK_MY;
    const size_t NUM_BLOCK = max_blocks < tmp3 ? max_blocks : tmp3;

    unsigned int *d_count_precision;                                                                  // GPu上的存储，不同精度的数量
    unsigned int *h_count_precision = (unsigned int *)malloc(sizeof(unsigned int) * NUM_BLOCK * THREADS_PER_BLOCK_MY * 3); // CPU上的存储
    cudaMalloc(((void **)(&d_count_precision)), NUM_BLOCK * THREADS_PER_BLOCK_MY * 3 * sizeof(unsigned int));
    
    const int num_rows_per_block = (((A->num_rows + NUM_BLOCK - 1) / NUM_BLOCK + THREADS_PER_BLOCK_MY - 1) / THREADS_PER_BLOCK_MY) * THREADS_PER_BLOCK_MY;
    
    // cusp::csr_matrix<int,float,cusp::host_memory> B(A);
    SpM B1;
    B1.rows = (unsigned int*)malloc(sizeof(unsigned int)*(A->num_rows+1));
    B1.cols = (unsigned int*)malloc(sizeof(unsigned int)*A->num_entries);
    B1.vals = (double*)malloc(sizeof(double)*A->num_entries);  
    for (unsigned int i = 0; i < A->num_rows + 1; i++)
    {
        B1.rows[i]=A->row_offsets[i];
    }
    for(unsigned int p=0;p<A->num_entries;p++){
        B1.vals[p]=A->values[p];
        B1.cols[p]=A->column_indices[p];
    }
    unsigned int *d_rowPtr;
    unsigned int *d_colInd;
    double *d_vals;
    cudaMalloc(((void **)(&d_rowPtr)), (A->num_rows + 1) * sizeof(int));
    cudaMalloc(((void **)(&d_colInd)), A->num_entries * sizeof(int));
    cudaMalloc(((void **)(&d_vals)), A->num_entries * sizeof(double));
    cudaMemcpy(d_rowPtr, B1.rows, (A->num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd, B1.cols, A->num_entries * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, B1.vals, A->num_entries * sizeof(double), cudaMemcpyHostToDevice);
    count_GPU<<<NUM_BLOCK, THREADS_PER_BLOCK_MY>>>(A->num_rows, num_rows_per_block, d_rowPtr, d_vals, d_count_precision);

    cudaMemcpy(h_count_precision, d_count_precision, NUM_BLOCK * THREADS_PER_BLOCK_MY * 3 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    int num_rows_per_thread = (num_rows_per_block + THREADS_PER_BLOCK_MY - 1) / THREADS_PER_BLOCK_MY;
    for (unsigned int i = 0; i < NUM_BLOCK * THREADS_PER_BLOCK_MY && i * num_rows_per_thread < A->num_rows; i++)
    {

        unsigned int tmph = csrh_nnz;
        unsigned int tmps = csrs_nnz;
        unsigned int tmpd = csrd_nnz;
        csrh_nnz += h_count_precision[i * 3];
        csrs_nnz += h_count_precision[i * 3 + 1];
        csrd_nnz += h_count_precision[i * 3 + 2];
        h_count_precision[i * 3] = tmph;
        h_count_precision[i * 3 + 1] = tmps;
        h_count_precision[i * 3 + 2] = tmpd;
    }
    cudaMemcpy(d_count_precision, h_count_precision, NUM_BLOCK * THREADS_PER_BLOCK_MY * 3 * sizeof(int), cudaMemcpyHostToDevice);
    AH->ncols = A->num_cols;
    AH->nrows = A->num_rows;
    AH->nnz = csrh_nnz;

    cudaMalloc(((void **)(&AH->rows)), sizeof(unsigned int) * (AH->nrows + 1));
    cudaMalloc(((void **)(&AH->cols)), csrh_nnz * sizeof(unsigned int));
    cudaMalloc(((void **)(&AH->vals)), csrh_nnz * sizeof(half));
    /**********************HSD——Single   分配空间CPU***************************/
    AS->ncols = A->num_cols;
    AS->nrows = A->num_rows;
    AS->nnz = csrs_nnz;

    cudaMalloc(((void **)(&AS->rows)), sizeof(unsigned int) * (AS->nrows + 1));
    cudaMalloc(((void **)(&AS->cols)), csrs_nnz * sizeof(unsigned int));
    cudaMalloc(((void **)(&AS->vals)), csrs_nnz * sizeof(float));
    /***********************HSD——double   分配空间CPU***************************/
    AD->ncols = A->num_cols;
    AD->nrows = A->num_rows;
    AD->nnz = csrd_nnz;

    cudaMalloc(((void **)(&AD->rows)), sizeof(unsigned int) * (AD->nrows + 1));
    cudaMalloc(((void **)(&AD->cols)), csrd_nnz * sizeof(unsigned int));
    cudaMalloc(((void **)(&AD->vals)), csrd_nnz * sizeof(double));

    split_GPU<<<NUM_BLOCK, THREADS_PER_BLOCK_MY>>>(A->num_rows, num_rows_per_block, d_rowPtr, d_colInd, d_vals, d_count_precision, AH->rows, AH->cols, AH->vals, AS->rows, AS->cols, AS->vals, AD->rows, AD->cols, AD->vals);
    free(h_count_precision);
    cudaFree(d_count_precision);
    cudaFree(d_rowPtr);
    cudaFree(d_colInd);
    cudaFree(d_vals);
    free(B1.rows);
    free(B1.cols);
    free(B1.vals);
}

int cal_vectors1(int sqrt_avg)
{

    int i;
    for (i = 2; i <= 32; i = i << 1) // half
    {
        if (sqrt_avg <= i)
        {
            return i;
        }
        else if (i == 32)
        {
            return 32;
        }
    }
    return 2;
}


void mixed_spmv(SpMH *AH, SpMS *AS, SpM *AD, float *d_xss, double *d_xd, double *d_y_HSD)
{
    size_t MAX_BLOCKS_HSD = 0;
    // const int THREADS_PER_BLOCK_MY = 512;
    int csrd_nnz = AD->nnz;
    int csrs_nnz = AS->nnz;
    int csrh_nnz = AH->nnz;
 
    int max = 2;

    // if (csrd_nnz == 0 && csrs_nnz && csrh_nnz) //半精度+单精度
    // {
    //     if (max == 2)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HS<2>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else if (max == 4)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HS<4>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else if (max == 8)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HS<8>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else if (max == 16)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HS<16>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HS<32>, THREADS_PER_BLOCK_MY, 0);
    //     }
    // }
    // else if (csrd_nnz && csrs_nnz == 0 && csrh_nnz) //半精度+双精度
    // {

    //     if (max == 2)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HD<2>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else if (max == 4)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HD<4>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else if (max == 8)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HD<8>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else if (max == 16)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HD<16>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HD<32>, THREADS_PER_BLOCK_MY, 0);
    //     }
    // }
    // else if (csrd_nnz && csrs_nnz && csrh_nnz == 0) //单精度+双精度
    // {
    //     if (max == 2)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_SD<2>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else if (max == 4)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_SD<4>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else if (max == 8)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_SD<8>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else if (max == 16)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_SD<16>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_SD<32>, THREADS_PER_BLOCK_MY, 0);
    //     }
    // }
    // else if (csrd_nnz && csrs_nnz && csrh_nnz) //半精度+单精度+双精度
    // {
    //     if (max == 2)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HSD<2>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else if (max == 4)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HSD<4>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else if (max == 8)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HSD<8>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else if (max == 16)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HSD<16>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HSD<32>, THREADS_PER_BLOCK_MY, 0);
    //     }
    // }
    // else if (csrd_nnz == 0 && csrs_nnz == 0 && csrh_nnz) //半精度
    // {
    //     if (max == 2)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_Hs<2>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else if (max == 4)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_Hs<4>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else if (max == 8)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_Hs<8>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else if (max == 16)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_Hs<16>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_Hs<32>, THREADS_PER_BLOCK_MY, 0);
    //     }
    // }
    // else if (csrd_nnz == 0 && csrs_nnz && csrh_nnz == 0) //单精度
    {
        // if (max == 2)
        // {
            MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_S<2>, THREADS_PER_BLOCK_MY, 0);
        // }
        // else if (max == 4)
        // {
        //     MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_S<4>, THREADS_PER_BLOCK_MY, 0);
        // }
        // else if (max == 8)
        // {
        //     MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_S<8>, THREADS_PER_BLOCK_MY, 0);
        // }
        // else if (max == 16)
        // {
        //     MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_S<16>, THREADS_PER_BLOCK_MY, 0);
        // }
        // else
        // {
        //     MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_S<32>, THREADS_PER_BLOCK_MY, 0);
        // }
    }
    // else //双精度
    // {
    //     if (max == 2)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<2>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else if (max == 4)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<4>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else if (max == 8)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<8>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else if (max == 16)
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<16>, THREADS_PER_BLOCK_MY, 0);
    //     }
    //     else
    //     {
    //         MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<32>, THREADS_PER_BLOCK_MY, 0);
    //     }
    // }

    const size_t VECTORS_PER_BLOCK_HSD = THREADS_PER_BLOCK_MY / max; //每个block中有多少个vectored threads
    int tmp = (AS->nrows + (VECTORS_PER_BLOCK_HSD - 1)) / VECTORS_PER_BLOCK_HSD;
    int min_num = MAX_BLOCKS_HSD < tmp ? MAX_BLOCKS_HSD : tmp;
    const size_t NUM_BLOCKS_HSD = min_num < 1 ? 1 : min_num;

    // if (csrh_nnz && csrs_nnz && csrd_nnz == 0) //半精度+单精度
    // {
    //     if (max == 2)
    //     {
    //         spmv_GPU_HS<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, d_xss, d_xd, d_y_HSD, AS->vals, AS->cols, AS->rows, AH->vals, AH->cols, AH->rows);
    //     }
    //     else if (max == 4)
    //     {
    //         spmv_GPU_HS<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, d_xss, d_xd, d_y_HSD, AS->vals, AS->cols, AS->rows, AH->vals, AH->cols, AH->rows);
    //     }
    //     else if (max == 8)
    //     {
    //         spmv_GPU_HS<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, d_xss, d_xd, d_y_HSD, AS->vals, AS->cols, AS->rows, AH->vals, AH->cols, AH->rows);
    //     }
    //     else if (max == 16)
    //     {
    //         spmv_GPU_HS<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, d_xss, d_xd, d_y_HSD, AS->vals, AS->cols, AS->rows, AH->vals, AH->cols, AH->rows);
    //     }
    //     else
    //     {
    //         spmv_GPU_HS<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, d_xss, d_xd, d_y_HSD, AS->vals, AS->cols, AS->rows, AH->vals, AH->cols, AH->rows);
    //     }
    // }
    // else if (csrh_nnz && csrs_nnz == 0 && csrd_nnz) //半精度+双精度
    // {
    //     if (max == 2)
    //     {
    //         spmv_GPU_HD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AH->nrows, d_xss, d_xd, d_y_HSD, AD->vals, AD->cols, AD->rows, AH->vals, AH->cols, AH->rows);
    //     }
    //     else if (max == 4)
    //     {
    //         spmv_GPU_HD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AH->nrows, d_xss, d_xd, d_y_HSD, AD->vals, AD->cols, AD->rows, AH->vals, AH->cols, AH->rows);
    //     }
    //     else if (max == 8)
    //     {
    //         spmv_GPU_HD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AH->nrows, d_xss, d_xd, d_y_HSD, AD->vals, AD->cols, AD->rows, AH->vals, AH->cols, AH->rows);
    //     }
    //     else if (max == 16)
    //     {
    //         spmv_GPU_HD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AH->nrows, d_xss, d_xd, d_y_HSD, AD->vals, AD->cols, AD->rows, AH->vals, AH->cols, AH->rows);
    //     }
    //     else
    //     {
    //         spmv_GPU_HD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AH->nrows, d_xss, d_xd, d_y_HSD, AD->vals, AD->cols, AD->rows, AH->vals, AH->cols, AH->rows);
    //     }
    // }
    // else if (csrh_nnz == 0 && csrs_nnz && csrd_nnz) //单精度+双精度
    // {
    //     if (max == 2)
    //     {
    //         spmv_GPU_SD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, d_xd, d_y_HSD, AS->vals, AS->cols, AS->rows, AD->vals, AD->cols, AD->rows);
    //     }
    //     else if (max == 4)
    //     {
    //         spmv_GPU_SD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, d_xd, d_y_HSD, AS->vals, AS->cols, AS->rows, AD->vals, AD->cols, AD->rows);
    //     }
    //     else if (max == 8)
    //     {
    //         spmv_GPU_SD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, d_xd, d_y_HSD, AS->vals, AS->cols, AS->rows, AD->vals, AD->cols, AD->rows);
    //     }
    //     else if (max == 16)
    //     {
    //         spmv_GPU_SD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, d_xd, d_y_HSD, AS->vals, AS->cols, AS->rows, AD->vals, AD->cols, AD->rows);
    //     }
    //     else
    //     {
    //         spmv_GPU_SD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, d_xd, d_y_HSD, AS->vals, AS->cols, AS->rows, AD->vals, AD->cols, AD->rows);
    //     }
    // }
    // else if (csrh_nnz && csrs_nnz && csrd_nnz) //半精度+单精度+双精度
    // {
    //     if (max == 2)
    //     {
    //         spmv_GPU_HSD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, d_xss, d_xd, d_y_HSD, AS->vals, AS->cols, AS->rows, AD->vals, AD->cols, AD->rows, AH->vals, AH->cols, AH->rows);
    //     }
    //     else if (max == 4)
    //     {
    //         spmv_GPU_HSD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, d_xss, d_xd, d_y_HSD, AS->vals, AS->cols, AS->rows, AD->vals, AD->cols, AD->rows, AH->vals, AH->cols, AH->rows);
    //     }
    //     else if (max == 8)
    //     {
    //         spmv_GPU_HSD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, d_xss, d_xd, d_y_HSD, AS->vals, AS->cols, AS->rows, AD->vals, AD->cols, AD->rows, AH->vals, AH->cols, AH->rows);
    //     }
    //     else if (max == 16)
    //     {
    //         spmv_GPU_HSD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, d_xss, d_xd, d_y_HSD, AS->vals, AS->cols, AS->rows, AD->vals, AD->cols, AD->rows, AH->vals, AH->cols, AH->rows);
    //     }
    //     else
    //     {
    //         spmv_GPU_HSD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, d_xss, d_xd, d_y_HSD, AS->vals, AS->cols, AS->rows, AD->vals, AD->cols, AD->rows, AH->vals, AH->cols, AH->rows);
    //     }
    // }
    // else if (csrh_nnz && csrs_nnz == 0 && csrd_nnz == 0) //半精度
    // {
    //     if (max == 2)
    //     {
    //         spmv_GPU_Hs<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AH->nrows, d_xss, d_y_HSD, AH->vals, AH->cols, AH->rows);
    //     }
    //     else if (max == 4)
    //     {
    //         spmv_GPU_Hs<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AH->nrows, d_xss, d_y_HSD, AH->vals, AH->cols, AH->rows);
    //     }
    //     else if (max == 8)
    //     {
    //         spmv_GPU_Hs<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AH->nrows, d_xss, d_y_HSD, AH->vals, AH->cols, AH->rows);
    //     }
    //     else if (max == 16)
    //     {
    //         spmv_GPU_Hs<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AH->nrows, d_xss, d_y_HSD, AH->vals, AH->cols, AH->rows);
    //     }
    //     else
    //     {
    //         spmv_GPU_Hs<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AH->nrows, d_xss, d_y_HSD, AH->vals, AH->cols, AH->rows);
    //     }
    // }
    // else if (csrh_nnz == 0 && csrs_nnz && csrd_nnz == 0) //单精度
    {
        // if (max == 2)
        {
            spmv_GPU_S<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, AS->vals, d_xd, AS->cols, AS->rows, d_y_HSD);
        }
        // else if (max == 4)
        // {
        //     spmv_GPU_S<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, AS->vals, d_xd, AS->cols, AS->rows, d_y_HSD);
        // }
        // else if (max == 8)
        // {
        //     spmv_GPU_S<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, AS->vals, d_xd, AS->cols, AS->rows, d_y_HSD);
        // }
        // else if (max == 16)
        // {
        //     spmv_GPU_S<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, AS->vals, d_xd, AS->cols, AS->rows, d_y_HSD);
        // }
        // else
        // {
        //     spmv_GPU_S<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AS->nrows, AS->vals, d_xd, AS->cols, AS->rows, d_y_HSD);
        // }
    }
    // else //双精度
    // {
    //     if (max == 2)
    //     {
    //         spmv_GPU_D<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AD->nrows, d_xd, d_y_HSD, AD->vals, AD->cols, AD->rows);
    //     }
    //     else if (max == 4)
    //     {
    //         spmv_GPU_D<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AD->nrows, d_xd, d_y_HSD, AD->vals, AD->cols, AD->rows);
    //     }
    //     else if (max == 8)
    //     {
    //         spmv_GPU_D<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AD->nrows, d_xd, d_y_HSD, AD->vals, AD->cols, AD->rows);
    //     }
    //     else if (max == 16)
    //     {
    //         spmv_GPU_D<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AD->nrows, d_xd, d_y_HSD, AD->vals, AD->cols, AD->rows);
    //     }
    //     else
    //     {
    //         spmv_GPU_D<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK_MY, 0>>>(AD->nrows, d_xd, d_y_HSD, AD->vals, AD->cols, AD->rows);
    //     }
    // }
}
} // end namespace krylov
} // end namespace cusp

#include <cusp/krylov/detail/gmres.inl>

