#include "spmv.h"
#include <cusp/system/cuda/arch.h>
#include <math.h>

uint cal_vectors(uint sqrt_avg)
{

    uint i;
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

void spmv(SpM<double> *A, double *x, double *y)
{ // A x y均已在GPU上

    uint avg = (A->nnz + A->nrows - 1) / A->nrows;
    uint sqr = sqrt(avg);
    uint THREADS_PER_VECTORS = cal_vectors(sqr);

    size_t MAX_BLOCKS = 0;
    if (THREADS_PER_VECTORS == 2)
    {
        MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<2>, THREADS_PER_BLOCK_SPMV, 0);
    }
    else if (THREADS_PER_VECTORS == 4)
    {
        MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<4>, THREADS_PER_BLOCK_SPMV, 0);
    }
    else if (THREADS_PER_VECTORS == 8)
    {
        MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<8>, THREADS_PER_BLOCK_SPMV, 0);
    }
    else if (THREADS_PER_VECTORS == 16)
    {
        MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<16>, THREADS_PER_BLOCK_SPMV, 0);
    }
    else
    {
        MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<32>, THREADS_PER_BLOCK_SPMV, 0);
    }
    const size_t VECTORS_PER_BLOCKS = THREADS_PER_BLOCK_SPMV / THREADS_PER_VECTORS;
    const size_t tmp = A->nrows + (THREADS_PER_VECTORS - 1) / VECTORS_PER_BLOCKS;
    const size_t NUM_BLOCKSD = min(MAX_BLOCKS, tmp < 1 ? 1 : tmp);

    if (THREADS_PER_VECTORS == 2)
    {

        spmv_GPU_D<2><<<NUM_BLOCKSD, THREADS_PER_BLOCK_SPMV, 0>>>(A->nrows, x, y, A->vals, A->cols, A->rows);
    }
    else if (THREADS_PER_VECTORS == 4)
    {

        spmv_GPU_D<4><<<NUM_BLOCKSD, THREADS_PER_BLOCK_SPMV, 0>>>(A->nrows, x, y, A->vals, A->cols, A->rows);
    }
    else if (THREADS_PER_VECTORS == 8)
    {

        spmv_GPU_D<8><<<NUM_BLOCKSD, THREADS_PER_BLOCK_SPMV, 0>>>(A->nrows, x, y, A->vals, A->cols, A->rows);
    }
    else if (THREADS_PER_VECTORS == 16)
    {

        spmv_GPU_D<16><<<NUM_BLOCKSD, THREADS_PER_BLOCK_SPMV, 0>>>(A->nrows, x, y, A->vals, A->cols, A->rows);
    }
    else
    {
        spmv_GPU_D<32><<<NUM_BLOCKSD, THREADS_PER_BLOCK_SPMV, 0>>>(A->nrows, x, y, A->vals, A->cols, A->rows);
    }
}
