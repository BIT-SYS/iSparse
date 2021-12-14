#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cusp/system/cuda/arch.h>

#define THREADS_PER_BLOCK_NORM 512

__global__ void norm_GPU(uint n, double *a, double *out)
{
    __shared__ volatile double cache[THREADS_PER_BLOCK_NORM];

    uint thread_id = thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    uint cache_index = threadIdx.x;

    double tmp1 = 0.0;
    for (uint i = thread_id; i < n; i += blockDim.x * gridDim.x)
    {
        tmp1 += a[thread_id] * a[thread_id];
    }
    cache[cache_index] = tmp1;
    // if (cache_index == 0)
    // {
    //     printf("%f\n", cache[0]);
    // }
    // synchronize threads in this block
    __syncthreads();
    if (blockDim.x >= 512 && threadIdx.x < 256)
    {
        cache[threadIdx.x] += cache[threadIdx.x + 256];
        __syncthreads();
    }
    if (blockDim.x >= 256 && threadIdx.x < 128)
    {
        cache[threadIdx.x] += cache[threadIdx.x + 128];
        __syncthreads();
    }
    if (blockDim.x >= 128 && threadIdx.x < 64)
    {
        cache[threadIdx.x] += cache[threadIdx.x + 64];
        __syncthreads();
    }
    // unroll last warp no sync needed
    if (threadIdx.x < 32)
    {
        if (blockDim.x >= 64)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 32];
            __syncwarp();
        }
        if (blockDim.x >= 32)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 16];
            __syncwarp();
        }
        if (blockDim.x >= 16)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 8];
            __syncwarp();
        }
        if (blockDim.x >= 8)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 4];
            __syncwarp();
        }
        if (blockDim.x >= 4)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 2];
            __syncwarp();
        }
        if (blockDim.x >= 2)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 1];
        }
    }
    if (cache_index == 0)
    {
        // printf("%f\n", cache[0]);
        out[blockIdx.x] = cache[0];
    }
}

__global__ void reduce_GPU(double *a, double *norm_out)
{

    __shared__ volatile double cache[THREADS_PER_BLOCK_NORM]; // thread shared memory
    uint thread_id = thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    cache[threadIdx.x] = a[threadIdx.x];

    if (blockDim.x >= 512 && threadIdx.x < 256)
    {
        cache[threadIdx.x] += cache[threadIdx.x + 256];
        __syncthreads();
    }
    if (blockDim.x >= 256 && threadIdx.x < 128)
    {
        cache[threadIdx.x] += cache[threadIdx.x + 128];
        __syncthreads();
    }
    if (blockDim.x >= 128 && threadIdx.x < 64)
    {
        cache[threadIdx.x] += cache[threadIdx.x + 64];
        __syncthreads();
    }

    // unroll last warp no sync needed
    if (threadIdx.x < 32)
    {
        if (blockDim.x >= 64)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 32];
            __syncwarp();
        }
        if (blockDim.x >= 32)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 16];
            __syncwarp();
        }
        if (blockDim.x >= 16)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 8];
            __syncwarp();
        }
        if (blockDim.x >= 8)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 4];
            __syncwarp();
        }
        if (blockDim.x >= 4)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 2];
            __syncwarp();
        }
        if (blockDim.x >= 2)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 1];
        }
    }

    if (thread_id == 0)
    {
        // printf("%f\n", cache[0]);
        norm_out[0] = sqrt(cache[0]);
        // printf("%f\n", norm_out[0]);
    }
}

void norm2(uint n, double *a, double *ret)
{

    size_t MAX_BLOCKS = 0;
    MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(norm_GPU, THREADS_PER_BLOCK_NORM, 0);
    const size_t tmp = (n + (THREADS_PER_BLOCK_NORM - 1)) / THREADS_PER_BLOCK_NORM;
    const size_t NUM_BLOCKS = min(MAX_BLOCKS, tmp < 1 ? 1 : tmp);

    double *out;
    cudaMalloc((void **)&out, sizeof(double) * NUM_BLOCKS);
    norm_GPU<<<NUM_BLOCKS, THREADS_PER_BLOCK_NORM, 0>>>(n, a, out);

    reduce_GPU<<<1, THREADS_PER_BLOCK_NORM, 0>>>(out, ret);
    cudaFree(out);
}

__global__ void dot_GPU(uint n, double *a, double *b, double *out)
{
    __shared__ volatile double cache[THREADS_PER_BLOCK_NORM];

    uint thread_id = thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    uint cache_index = threadIdx.x;

    double tmp1 = 0.0;
    for (uint i = thread_id; i < n; i += blockDim.x * gridDim.x)
    {
        tmp1 += a[thread_id] * b[thread_id];
    }
    cache[cache_index] = tmp1;
    // if (cache_index == 0)
    // {
    //     printf("%f\n", cache[0]);
    // }
    // synchronize threads in this block
    __syncthreads();
    if (blockDim.x >= 512 && threadIdx.x < 256)
    {
        cache[threadIdx.x] += cache[threadIdx.x + 256];
        __syncthreads();
    }
    if (blockDim.x >= 256 && threadIdx.x < 128)
    {
        cache[threadIdx.x] += cache[threadIdx.x + 128];
        __syncthreads();
    }
    if (blockDim.x >= 128 && threadIdx.x < 64)
    {
        cache[threadIdx.x] += cache[threadIdx.x + 64];
        __syncthreads();
    }
    // unroll last warp no sync needed
    if (threadIdx.x < 32)
    {
        if (blockDim.x >= 64)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 32];
            __syncwarp();
        }
        if (blockDim.x >= 32)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 16];
            __syncwarp();
        }
        if (blockDim.x >= 16)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 8];
            __syncwarp();
        }
        if (blockDim.x >= 8)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 4];
            __syncwarp();
        }
        if (blockDim.x >= 4)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 2];
            __syncwarp();
        }
        if (blockDim.x >= 2)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 1];
        }
    }
    if (cache_index == 0)
    {
        // printf("%f\n", cache[0]);
        out[blockIdx.x] = cache[0];
    }
}

__global__ void reduce_GPU2(double *a, double *norm_out)
{

    __shared__ volatile double cache[THREADS_PER_BLOCK_NORM]; // thread shared memory
    uint thread_id = thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    cache[threadIdx.x] = a[threadIdx.x];

    if (blockDim.x >= 512 && threadIdx.x < 256)
    {
        cache[threadIdx.x] += cache[threadIdx.x + 256];
        __syncthreads();
    }
    if (blockDim.x >= 256 && threadIdx.x < 128)
    {
        cache[threadIdx.x] += cache[threadIdx.x + 128];
        __syncthreads();
    }
    if (blockDim.x >= 128 && threadIdx.x < 64)
    {
        cache[threadIdx.x] += cache[threadIdx.x + 64];
        __syncthreads();
    }

    // unroll last warp no sync needed
    if (threadIdx.x < 32)
    {
        if (blockDim.x >= 64)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 32];
            __syncwarp();
        }
        if (blockDim.x >= 32)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 16];
            __syncwarp();
        }
        if (blockDim.x >= 16)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 8];
            __syncwarp();
        }
        if (blockDim.x >= 8)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 4];
            __syncwarp();
        }
        if (blockDim.x >= 4)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 2];
            __syncwarp();
        }
        if (blockDim.x >= 2)
        {
            cache[threadIdx.x] += cache[threadIdx.x + 1];
        }
    }

    if (thread_id == 0)
    {
        // printf("%f\n", cache[0]);
        norm_out[0] = (cache[0]);
        // printf("%f\n", norm_out[0]);
    }
}

void dot2(uint n, double *a, double *b, double *ret)
{

    size_t MAX_BLOCKS = 0;
    MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(dot_GPU, THREADS_PER_BLOCK_NORM, 0);
    const size_t tmp = (n + (THREADS_PER_BLOCK_NORM - 1)) / THREADS_PER_BLOCK_NORM;
    const size_t NUM_BLOCKS = min(MAX_BLOCKS, tmp < 1 ? 1 : tmp);

    double *out;
    cudaMalloc((void **)&out, sizeof(double) * NUM_BLOCKS);
    dot_GPU<<<NUM_BLOCKS, THREADS_PER_BLOCK_NORM, 0>>>(n, a, b, out);

    reduce_GPU2<<<1, THREADS_PER_BLOCK_NORM, 0>>>(out, ret);
    cudaFree(out);
}