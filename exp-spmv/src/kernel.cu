#include "kernel.h"

__global__ void count_GPU(
    uint num_rows,
    uint num_rows_per_block,
    int exp,
    uint *rowPtr,
    double *vals,
    uint *count)
{
    const uint thread_id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
    const uint num_rows_per_thread = (num_rows_per_block + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    uint begin = num_rows_per_block * blockIdx.x + threadIdx.x * num_rows_per_thread;
    uint end = num_rows_per_block * blockIdx.x + (threadIdx.x + 1) * num_rows_per_thread;
    if (end >= num_rows)
    {
        end = num_rows;
    }

    uint count_s = 0;
    uint count_d = 0;

    int exponent;
    int32_t *halfval;

    for (uint i = begin; i < end; i++)
    {
        for (uint j = rowPtr[i]; j < rowPtr[i + 1]; j++)
        {
            halfval = (int32_t *)(vals + j);
            exponent = ((halfval[1] >> 20) & 0x7ff) - 1023;

            if (exponent == exp&&(halfval[0]&0x1ffff)==0)
            {
                count_s++; //单精度
            }
            else
            {
                count_d++; //双精度
            }
        }
    }
    count[thread_id * 2] = count_s;
    count[thread_id * 2 + 1] = count_d;
}

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
    double *valD)
{
    const uint thread_id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
    const uint num_rows_per_thread = (num_rows_per_block + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // const uint block_edge = num_rows_per_block * (blockIdx.x + 1);
    uint begin = num_rows_per_block * blockIdx.x + threadIdx.x * num_rows_per_thread;
    uint end = num_rows_per_block * blockIdx.x + (threadIdx.x + 1) * num_rows_per_thread;
    if (end >= num_rows)
    {
        end = num_rows;
    }

    uint count_s = count[thread_id * 2 + 0];
    uint count_d = count[thread_id * 2 + 1];

    int exponent;
    int32_t *halfval;

    for (uint i = begin; i < end; i++)
    {

        rowPtrS[i] = count_s;
        rowPtrD[i] = count_d;
        for (uint j = rowPtr[i]; j < rowPtr[i + 1]; j++)
        {
            halfval = (int32_t *)(vals + j);
            exponent = ((halfval[1] >> 20) & 0x7ff) - 1023;
            if (exponent == exp&&(halfval[0]&0x1ffff)==0)
            {
                colIndS[count_s] = colInd[j];
                // valS[count_s] = vals[j];
                int32_t tran_tmp = halfval[1]&0x80000000;
                tran_tmp = tran_tmp|((halfval[1]&0xfffff)<<11);
                tran_tmp = tran_tmp | ((halfval[0]&0xffe00000)>>21);
                valS[count_s]=tran_tmp;
                count_s++; //单精度
            }
            else
            {

                colIndD[count_d] = colInd[j];
                valD[count_d] = vals[j];
                count_d++; //双精度
            }
        }
    }

    if (end == num_rows && begin < num_rows)
    {
        rowPtrS[num_rows] = count_s;
        rowPtrD[num_rows] = count_d;
    }
}
