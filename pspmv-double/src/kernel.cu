#include "kernel.h"

__global__ void count_GPU(
    uint num_rows,
    uint num_rows_per_block,
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

    uint count_h = 0;
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

            if (vals[j] == 0)
            {
                count_h++; //半精度
            }
            else if (exponent >= -15 && exponent <= 15)
            {
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
}

__global__ void split_GPU(
    uint num_rows,
    uint num_rows_per_block,
    uint *rowPtr,
    uint *colInd,
    double *vals,
    uint *count,
    uint *rowPtrH,
    uint *colIndH,
    half *valH,
    uint *rowPtrS,
    uint *colIndS,
    float *valS,
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

    uint count_h = count[thread_id * 3];
    uint count_s = count[thread_id * 3 + 1];
    uint count_d = count[thread_id * 3 + 2];

    int exponent;
    int32_t *halfval;

    for (uint i = begin; i < end; i++)
    {

        rowPtrH[i] = count_h;
        rowPtrS[i] = count_s;
        rowPtrD[i] = count_d;
        for (uint j = rowPtr[i]; j < rowPtr[i + 1]; j++)
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
                    count_d++; //双精度
                }
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
        rowPtrH[num_rows] = count_h;
        rowPtrS[num_rows] = count_s;
        rowPtrD[num_rows] = count_d;
    }
}
