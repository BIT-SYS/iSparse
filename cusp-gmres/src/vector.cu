#include <cusp/system/cuda/arch.h>
#include <math.h>
#include "sparseMatrix.h"
#include <cuda_runtime.h>
#define THREADS_PER_BLOCK_VECTOR 512

// a=a+b*s
__global__ void axpy_GPU(
    uint Am,
    double *a,
    double *b,
    double s)
{

    __shared__ volatile double tempy[THREADS_PER_BLOCK_VECTOR];

    const uint thread_id = THREADS_PER_BLOCK_VECTOR * blockIdx.x + threadIdx.x;

    const uint num_vectors = THREADS_PER_BLOCK_VECTOR * gridDim.x;

    for (uint row = thread_id; row < Am; row += num_vectors)
    {
        tempy[threadIdx.x] = a[row] + b[row] * s;
        b[row] = tempy[threadIdx.x];
    }
}

/*
*********rank(a)=Am
*********a=a-b*s
*/
void axpy2(uint Am, double *a, double *b, double s)
{
   
    size_t MAX_BLOCKS = 0;

    MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(axpy_GPU, THREADS_PER_BLOCK_VECTOR, 0);
    const size_t tmp = (Am + (THREADS_PER_BLOCK_VECTOR - 1)) / THREADS_PER_BLOCK_VECTOR;
    const size_t NUM_BLOCKS = min(MAX_BLOCKS, tmp < 1 ? 1 : tmp);
   
    axpy_GPU<<<NUM_BLOCKS, THREADS_PER_BLOCK_VECTOR, 0>>>(Am, a, b, s);
}

// a=a*s
__global__ void scal2_GPU(
    uint Am,
    double sig,
    double *a,
    double *s)
{
    __shared__ volatile double tempy[THREADS_PER_BLOCK_VECTOR];

    const uint thread_id = THREADS_PER_BLOCK_VECTOR * blockIdx.x + threadIdx.x;

    const uint num_vectors = THREADS_PER_BLOCK_VECTOR * gridDim.x;
    double s_tmp = sig / s[0];

    for (uint row = thread_id; row < Am; row += num_vectors)
    {
        tempy[threadIdx.x] = a[row] * s_tmp;
        a[row] = tempy[threadIdx.x];
    }
}

void scal2(uint Am, double sig, double *a, double *s)
{

    size_t MAX_BLOCKS = 0;

    MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(scal2_GPU, THREADS_PER_BLOCK_VECTOR, 0);
    const size_t tmp = (Am + (THREADS_PER_BLOCK_VECTOR - 1)) / THREADS_PER_BLOCK_VECTOR;
    const size_t NUM_BLOCKS = min(MAX_BLOCKS, tmp < 1 ? 1 : tmp);

    scal2_GPU<<<NUM_BLOCKS, THREADS_PER_BLOCK_VECTOR, 0>>>(Am, sig, a, s);
}

// a=a*s
__global__ void copy_GPU(
    uint Am,
    double *a,
    double *b)
{
    __shared__ volatile double tempy[THREADS_PER_BLOCK_VECTOR];
    const uint thread_id = THREADS_PER_BLOCK_VECTOR * blockIdx.x + threadIdx.x;

    const uint num_vectors = THREADS_PER_BLOCK_VECTOR * gridDim.x;

    for (uint row = thread_id; row < Am; row += num_vectors)
    {
        tempy[threadIdx.x] = b[row];
        a[row] = tempy[threadIdx.x];
    }
}

void copy2(uint Am, double *a, double *b)
{
    size_t MAX_BLOCKS = 0;

    MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(copy_GPU, THREADS_PER_BLOCK_VECTOR, 0);
    const size_t tmp = (Am + (THREADS_PER_BLOCK_VECTOR - 1)) / THREADS_PER_BLOCK_VECTOR;
    const size_t NUM_BLOCKS = min(MAX_BLOCKS, tmp < 1 ? 1 : tmp);

    copy_GPU<<<NUM_BLOCKS, THREADS_PER_BLOCK_VECTOR, 0>>>(Am, a, b);
}

__global__ void fill_GPU(
    uint Am,
    double *a,
    double s)
{
    __shared__ volatile double tempy[THREADS_PER_BLOCK_VECTOR];
    const uint thread_id = THREADS_PER_BLOCK_VECTOR * blockIdx.x + threadIdx.x;

    const uint num_vectors = THREADS_PER_BLOCK_VECTOR * gridDim.x;

    for (uint row = thread_id; row < Am; row += num_vectors)
    {
        tempy[threadIdx.x] = s;
        a[row] = tempy[threadIdx.x];
    }
}

void fill2(uint Am, double *a, double s)
{

    size_t MAX_BLOCKS = 0;

    MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(fill_GPU, THREADS_PER_BLOCK_VECTOR, 0);
    const size_t tmp = (Am + (THREADS_PER_BLOCK_VECTOR - 1)) / THREADS_PER_BLOCK_VECTOR;
    const size_t NUM_BLOCKS = min(MAX_BLOCKS, tmp < 1 ? 1 : tmp);

    fill_GPU<<<NUM_BLOCKS, THREADS_PER_BLOCK_VECTOR, 0>>>(Am, a, s);
}

__global__ void assign_GPU(
    double *a,
    double *s)
{

    const uint thread_id = THREADS_PER_BLOCK_VECTOR * blockIdx.x + threadIdx.x;
    if (thread_id == 0)
    {
        *a = *s;
    }
}

void assign2(double *a, double *s)
{
  
    const size_t NUM_BLOCKS = 1;
    assign_GPU<<<1, 1, 0>>>(a, s);
}