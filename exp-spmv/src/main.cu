/*
pSpMV源代码
*/

#include <cusp/system/cuda/arch.h>
#include "format.h"
#include "cusp/io/matrix_market.h"
#include "kernel.h"
#include <math.h>
#include <cusparse.h>
#include <map>
#include <vector>
#include <iostream>
using namespace std;
int NUM_ITERATIONS;

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) < (b) ? (b) : (a))

extern void Mp_SpMV(SpM *A, float *xs, double *xd, double *y_Mp, double *split_time_Mp, double *transfer_time_Mp, double *SpMV_time_Mp, uint *nnzs, uint *nnzd, int *vec);
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
bool cmp(const pair<int, int> left, const pair<int, int> right)
{
    return left.second > right.second;
}
void sortMapByValue(map<int, uint> &tMap, vector<pair<int, uint>> &tVector)
{
    for (map<int, uint>::iterator curr = tMap.begin(); curr != tMap.end();)
    {
        tVector.push_back(make_pair(curr->first, curr->second));
        tMap.erase(curr++);
    }

    sort(tVector.begin(), tVector.end(), cmp);
}
void statistics_D(SpM *csr, uint *min, uint *max, uint *nums, double *variance, double avg)
{
    uint len = 0; // csr->nrows * PERCENT;
    (*max) = 0;
    (*nums) = 0;
    (*min) = INT_MAX;
    for (uint i = 0; i < csr->nrows; i++)
    {
        uint nnz = csr->rows[i + 1] - csr->rows[i];
        (*variance) += ((double)nnz - avg) * ((double)nnz - avg);
        if ((*max) < nnz)
        {
            (*max) = nnz;
        }
        if (nnz > len)
        {
            (*nums) = nnz;
            len = nnz;
        }
        if (nnz != 0 && nnz < (*min))
        {
            (*min) = nnz;
        }
    }
}

void statistics_S(SpMS *csr, uint *min, uint *max, uint *nums, double *variance, double avg)
{
    uint len = 0; // csr->nrows * PERCENT;
    (*max) = 0;
    (*nums) = 0;
    (*min) = INT_MAX;
    for (uint i = 0; i < csr->nrows; i++)
    {
        uint nnz = csr->rows[i + 1] - csr->rows[i];
        (*variance) += ((double)nnz - avg) * ((double)nnz - avg);
        if ((*max) < nnz)
        {
            (*max) = nnz;
        }
        if (nnz > len)
        {
            (*nums) = nnz;
            len = nnz;
        }
        if (nnz != 0 && nnz < (*min))
        {
            (*min) = nnz;
        }
    }
}

void sample(SpM *A, int &exp)
{

    uint N = A->nrows * 0.05;

    if (N < 1)
    {
        N = 1;
    }

    uint step = A->nrows / N;
    map<int, uint> m;

    uint nnz = 0;
    uint k = 0;
    while (nnz == 0 && k < A->nrows)
    {
        for (uint i = k; i < A->nrows; i = i + step)
        {
            for (int j = A->rows[i]; j < A->rows[i + 1]; j++)
            {
                nnz++;
                int32_t *halfval = (int32_t *)(A->vals + j);
                int exponent = ((halfval[1] >> 20) & 0x7ff) - 1023;
                m[exponent]++;
            }
        }
        k++;
    }
    vector<pair<int, uint>> tVector;
    sortMapByValue(m, tVector);
    exp = tVector[0].first;
}

int main(int argc, char *argv[])
{
    // Device Initialization
    // Support for one GPU as of now
    // int deviceCount;
    // cudaGetDeviceCount(&deviceCount);
    // printf("cuda device count: %d\n", deviceCount);
    // if (deviceCount == 0)
    // {
    //     printf("No device supporting CUDA\n");
    //     exit(-1);
    // }
    cudaSetDevice(1);
    NUM_ITERATIONS = atoi(argv[3]);
    int max = atoi(argv[4]);

    uint i;
    float split_time = 0;
    // original matrix and three precision submatrices
    SpM A;
    SpM AD;
    SpMS AS;

    // load the matrix in CSR format
    A.readMtx(argv[1]);
    printf("%u  %u  %u\n", A.nrows, A.ncols, A.nnz);
    map<int, uint> m;
    for (i = 0; i < A.nnz; i++)
    {
        int32_t *halfval = (int32_t *)(A.vals + i);
        int exponent = ((halfval[1] >> 20) & 0x7ff) - 1023;
        m[exponent]++;
    }
    vector<pair<int, uint>> tVector;
    sortMapByValue(m, tVector);
    int exponent = tVector[0].first;
    vector<pair<int, uint>>().swap(tVector);
    printf("exponent:%d\n",exponent);

    cudaEvent_t start_event, stop_event;
    float cuda_elapsed_timed;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // int exponent;
    // cudaEventRecord(start_event, 0);
    // sample(&A, exponent);
    // cudaEventRecord(stop_event, 0);
    // cudaEventSynchronize(stop_event);
    // cudaEventElapsedTime(&cuda_elapsed_timed, start_event, stop_event);
    // FILE *f = fopen(argv[2], "a+");
    // fprintf(f, "%s\t%f\n",argv[1],cuda_elapsed_timed);
    // return 0;
    // FILE *fp1 = fopen(argv[2], "a+");
    // fprintf(fp1, "%s\t%d\n", argv[1], exponent);
    // fclose(fp1);
    // return 0;
    // object to record time

    // double precision vector x
    double *xd;

    // result vector y
    double *y_D;
    double *y_SD;

    // nnz in submatrix
    size_t csrs_nnz = 0;
    size_t csrd_nnz = 0;

    // device memory management
    uint *d_rowPtr;
    uint *d_colInd;
    double *d_vals;
    cudaMalloc(((void **)(&d_rowPtr)), (A.nrows + 1) * sizeof(uint));
    cudaMalloc(((void **)(&d_colInd)), A.nnz * sizeof(uint));
    cudaMalloc(((void **)(&d_vals)), A.nnz * sizeof(double));

    // part of transfer time —— transfer original matrix
    double transfer_time_HSD = 0;
    cudaEventRecord(start_event, 0);
    cudaMemcpy(d_rowPtr, A.rows, (A.nrows + 1) * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd, A.cols, A.nnz * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, A.vals, A.nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&cuda_elapsed_timed, start_event, stop_event);
    transfer_time_HSD += cuda_elapsed_timed / NUM_ITERATIONS;

    // min num of blocks between gpu kernel split_GPU and count_GPU
    uint max_blocks = min(cusp::system::cuda::detail::max_active_blocks(split_GPU, THREADS_PER_BLOCK, 0),
                          cusp::system::cuda::detail::max_active_blocks(count_GPU, THREADS_PER_BLOCK, 0));

    size_t NUM_BLOCK = min(max_blocks, (A.nrows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    const uint num_rows_per_block = (((A.nrows + NUM_BLOCK - 1) / NUM_BLOCK + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * THREADS_PER_BLOCK;
    NUM_BLOCK = min(NUM_BLOCK, (A.nrows + num_rows_per_block - 1) / num_rows_per_block);

    // record nnz for every thread in GPU
    uint *d_count_precision;
    uint *h_count_precision = (uint *)malloc(sizeof(uint) * NUM_BLOCK * THREADS_PER_BLOCK * 2);
    cudaMalloc(((void **)(&d_count_precision)), NUM_BLOCK * THREADS_PER_BLOCK * 2 * sizeof(uint));

    uint num_rows_per_thread = (num_rows_per_block + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // preprocess——count
    cudaEventRecord(start_event, 0);
    count_GPU<<<NUM_BLOCK, THREADS_PER_BLOCK>>>(A.nrows, num_rows_per_block, exponent, d_rowPtr, d_vals, d_count_precision);
    cudaMemcpy(h_count_precision, d_count_precision, NUM_BLOCK * THREADS_PER_BLOCK * 2 * sizeof(uint), cudaMemcpyDeviceToHost);
    // compute where to write for each thread
    for (uint i = 0; i < NUM_BLOCK * THREADS_PER_BLOCK && i * num_rows_per_thread < A.nrows; i++)
    {
        uint tmps = csrs_nnz;
        uint tmpd = csrd_nnz;
        csrs_nnz += h_count_precision[i * 2];
        csrd_nnz += h_count_precision[i * 2 + 1];
        h_count_precision[i * 2] = tmps;
        h_count_precision[i * 2 + 1] = tmpd;
    }
    cudaMemcpy(d_count_precision, h_count_precision, NUM_BLOCK * THREADS_PER_BLOCK * 2 * sizeof(uint), cudaMemcpyHostToDevice);
    // printf("nnz:%u  %u\n", csrs_nnz, csrd_nnz);
    // memory management
    AS.ncols = A.ncols;
    AS.nrows = A.nrows;
    AS.nnz = csrs_nnz;

    AD.ncols = A.ncols;
    AD.nrows = A.nrows;
    AD.nnz = csrd_nnz;

    double *d_xd;

    uint *d_csrRowPtrS;
    uint *d_csrColIndS;
    int32_t *d_csrValS;

    uint *d_csrRowPtrD;
    uint *d_csrColIndD;
    double *d_csrValD;

    double *d_y_D;
    double *d_y_SD;

    cudaMalloc(((void **)(&d_xd)), A.ncols * sizeof(double));

    cudaMalloc(((void **)(&d_y_SD)), A.nrows * sizeof(double));
    cudaMalloc(((void **)(&d_y_D)), A.nrows * sizeof(double));

    cudaMalloc(((void **)(&d_csrValS)), AS.nnz * sizeof(int32_t));
    cudaMalloc(((void **)(&d_csrColIndS)), AS.nnz * sizeof(uint));
    cudaMalloc(((void **)(&d_csrRowPtrS)), (A.nrows + 1) * sizeof(uint));

    cudaMalloc(((void **)(&d_csrValD)), AD.nnz * sizeof(double));
    cudaMalloc(((void **)(&d_csrColIndD)), AD.nnz * sizeof(uint));
    cudaMalloc(((void **)(&d_csrRowPtrD)), (A.nrows + 1) * sizeof(uint));

    // preprocess——split
    split_GPU<<<NUM_BLOCK, THREADS_PER_BLOCK>>>(A.nrows, num_rows_per_block, exponent, d_rowPtr, d_colInd, d_vals, d_count_precision, d_csrRowPtrS, d_csrColIndS, d_csrValS, d_csrRowPtrD, d_csrColIndD, d_csrValD);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&cuda_elapsed_timed, start_event, stop_event);

    split_time = cuda_elapsed_timed;

    cudaFree(d_vals);
    cudaFree(d_rowPtr);
    cudaFree(d_colInd);
    cudaFree(d_count_precision);
    free(h_count_precision);
    // printf("%d  %d\n",AS.nnz,AD.nnz);
    // return 0;
    // // printf("%s\n", argv[7]);
    // // return 0;
    // FILE *fp3 = fopen(argv[7], "a+");
    // if (fp3 == NULL)
    // {
    //     printf("failed open\n");
    // }
    // fprintf(fp3, "%s\n", "%%MatrixMarket matrix coordinate real general");
    // fprintf(fp3, "\t%d\t%d\t%d\n", A.nrows, A.ncols, A.nnz);
    // for (int i = 0; i < A.nrows; i++)
    // {
    //     for (int j = A.rows[i]; j < A.rows[i + 1]; j++)
    //     {
    //         fprintf(fp3, "%d %d %f\n", i+1, A.cols[j]+1, 1.0);
    //     }
    // }
    // fprintf(fp3, "\n");
    // fclose(fp3);
    // // free(A.cols);
    // // free(A.rows);
    // // free(A.vals);
    // // return 0;

    // AS.rows = (uint *)malloc(sizeof(uint) * (AS.nrows + 1));
    // AD.rows = (uint *)malloc(sizeof(uint) * (AD.nrows + 1));
    // cudaMemcpy(AS.rows, d_csrRowPtrS, sizeof(uint) * (AS.nrows + 1), cudaMemcpyDeviceToHost);
    // cudaMemcpy(AD.rows, d_csrRowPtrD, sizeof(uint) * (AD.nrows + 1), cudaMemcpyDeviceToHost);
    // AS.cols = (uint *)malloc(sizeof(uint) * (AS.nnz));
    // AD.cols = (uint *)malloc(sizeof(uint) * (AD.nnz));
    // cudaMemcpy(AS.cols, d_csrColIndS, sizeof(uint) * AS.nnz, cudaMemcpyDeviceToHost);
    // cudaMemcpy(AD.cols, d_csrColIndD, sizeof(uint) * AD.nnz, cudaMemcpyDeviceToHost);
    // FILE *fp1 = fopen(argv[5], "a+");
    // fprintf(fp1, "%s\n", "%%MatrixMarket matrix coordinate real general");
    // fprintf(fp1, "\t%d\t%d\t%d\n", AS.nrows, AS.ncols, AS.nnz);
    // for (int i = 0; i < AS.nrows; i++)
    // {
    //     for (int j = AS.rows[i]; j < AS.rows[i + 1]; j++)
    //     {
    //         fprintf(fp1, "%d %d %f\n", i+1, AS.cols[j]+1, 1.0);
    //     }
    // }
    // fprintf(fp1, "\n");
    // fclose(fp1);

    // FILE *fp2 = fopen(argv[6], "a+");
    // fprintf(fp2, "%s\n", "%%MatrixMarket matrix coordinate real general");
    // fprintf(fp2, "\t%d\t%d\t%d\n", AD.nrows, AD.ncols, AD.nnz);
    // for (int i = 0; i < AD.nrows; i++)
    // {
    //     for (int j = AD.rows[i]; j < AD.rows[i + 1]; j++)
    //     {
    //         fprintf(fp2, "%d %d %f\n", i+1, AD.cols[j]+1, 1.0);
    //     }
    // }
    // fprintf(fp2, "\n");
    // fclose(fp2);

    // return 0;

    // AS.rows = (uint *)malloc(sizeof(uint) * (AS.nrows + 1));
    // AD.rows = (uint *)malloc(sizeof(uint) * (AD.nrows + 1));
    // cudaMemcpy(AS.rows, d_csrRowPtrS, sizeof(uint) * (AS.nrows + 1), cudaMemcpyDeviceToHost);
    // cudaMemcpy(AD.rows, d_csrRowPtrD, sizeof(uint) * (AD.nrows + 1), cudaMemcpyDeviceToHost);
    // double avgs = AS.nnz / AS.nrows;
    // avgs = max(avgs, 1);
    // double avgd = max(AD.nnz / AD.nrows, 1);

    // uint mins = 0;
    // uint mind = 0;
    // uint maxs = 0;
    // uint maxd = 0;
    // uint nums = 0;
    // uint numd = 0;
    // double vars = 0;
    // double vard = 0;
    // if (csrs_nnz)
    //     statistics_S(&AS, &mins, &maxs, &nums, &vars, (double)(avgs));
    // if (csrd_nnz)
    //     statistics_D(&AD, &mind, &maxd, &numd, &vard, (double)(avgd));
    // FILE *fp1 = fopen(argv[2], "a+");
    // fprintf(fp1, "%s\t%u\t%u\t%u\t%u\t%u\t%f\t%u\t%u\t%u\t%f\t%u\t%u\t%u\t%f\t%f\n",
    //         argv[1],
    //         A.nrows, A.ncols, A.nnz, csrs_nnz, csrd_nnz,
    //         avgs, mins, maxs, nums,
    //         avgd, mind, maxd, numd,
    //         vars / (double)A.nrows, vard / (double)A.nrows);
    // fclose(fp1);
    // return 0;
    // printf("preprocess:  %u  %u\n", csrs_nnz, csrd_nnz);
    //-------------------------------------------------preprocess  finish----------------------------------------------------------------------------

    // initial vector x
    xd = ((double *)(malloc(sizeof(double) * A.ncols)));

    y_D = ((double *)(malloc(sizeof(double) * A.nrows)));
    y_SD = ((double *)(malloc(sizeof(double) * A.nrows)));

    srand(time(NULL));
    for (i = 0; i < A.ncols; i++)
    { //[0,1)浮点数
        // double tmp = (1 * rand() / (RAND_MAX + 1.0));
        // xs[i] = 1.0;
        xd[i] = 1.0;
        // xs[i] = (float)tmp;
        // xd[i] = tmp;
    }

    for (i = 0; i < A.nrows; i++)
    {
        y_D[i] = 0;
        y_SD[i] = 0;
    }

    uint avg = (A.nnz + A.nrows - 1) / A.nrows;
    uint sqr = sqrt(avg);
    uint THREADS_PER_VECTORS = cal_vectors(sqr);

    // part of transfertime——vector
    cudaEventRecord(start_event, 0);
    cudaMemcpy(d_xd, xd, sizeof(double) * A.ncols, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&cuda_elapsed_timed, start_event, stop_event);
    transfer_time_HSD += cuda_elapsed_timed;

    // int max;
    double SpMV_time_HSD[33];
    size_t MAX_BLOCKS_HSD = 0;

    // test five parameter
    // for (max = 2; max <= 32; max = max * 2)
    {
        // max = 8;
        if (csrd_nnz && csrs_nnz)
        { // single and double
            if (max == 2)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_SD<2>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 4)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_SD<4>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 8)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_SD<8>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 16)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_SD<16>, THREADS_PER_BLOCK, 0);
            }
            else
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_SD<32>, THREADS_PER_BLOCK, 0);
            }
        }

        else if (csrd_nnz == 0 && csrs_nnz)
        { // single
            if (max == 2)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_S<2>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 4)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_S<4>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 8)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_S<8>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 16)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_S<16>, THREADS_PER_BLOCK, 0);
            }
            else
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_S<32>, THREADS_PER_BLOCK, 0);
            }
        }
        else
        { // double
            if (max == 2)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<2>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 4)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<4>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 8)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<8>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 16)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<16>, THREADS_PER_BLOCK, 0);
            }
            else
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<32>, THREADS_PER_BLOCK, 0);
            }
        }

        const size_t VECTORS_PER_BLOCK_HSD = THREADS_PER_BLOCK / max;
        uint min_num = min(MAX_BLOCKS_HSD, (A.nrows + (VECTORS_PER_BLOCK_HSD - 1)) / VECTORS_PER_BLOCK_HSD);
        const size_t NUM_BLOCKS_HSD = min_num < 1 ? 1 : min_num;

        float time;
        cudaEventRecord(start_event, 0);
        // spmv start
        if (csrs_nnz && csrd_nnz)
        { // single and double
            if (max == 2)
            {
                spmv_GPU_SD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_xd, d_y_SD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
            }
            else if (max == 4)
            {
                spmv_GPU_SD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_xd, d_y_SD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
            }
            else if (max == 8)
            {
                spmv_GPU_SD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_xd, d_y_SD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
            }
            else if (max == 16)
            {
                spmv_GPU_SD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_xd, d_y_SD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
            }
            else
            {
                spmv_GPU_SD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_xd, d_y_SD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
            }
        }
        else if (csrs_nnz && csrd_nnz == 0)
        { // single
            if (max == 2)
            {
                spmv_GPU_S<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_SD);
            }
            else if (max == 4)
            {
                spmv_GPU_S<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_SD);
            }
            else if (max == 8)
            {
                spmv_GPU_S<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_SD);
            }
            else if (max == 16)
            {
                spmv_GPU_S<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_SD);
            }
            else
            {
                spmv_GPU_S<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_SD);
            }
        }
        else
        { // double
            if (max == 2)
            {
                spmv_GPU_D<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_SD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
            }
            else if (max == 4)
            {
                spmv_GPU_D<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_SD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
            }
            else if (max == 8)
            {
                spmv_GPU_D<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_SD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
            }
            else if (max == 16)
            {
                spmv_GPU_D<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_SD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
            }
            else
            {
                spmv_GPU_D<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_SD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
            }
        }

        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&cuda_elapsed_timed, start_event, stop_event); // ms

        // ms / 1000 = s
        time = cuda_elapsed_timed / 1000;

        if (time <= 0.000001)
        {
            NUM_ITERATIONS = NUM_ITERATIONS;
        }
        else
        {
            NUM_ITERATIONS = min(NUM_ITERATIONS, max(1, (size_t)(3 / time)));
        }

        cudaEventRecord(start_event, 0);
        for (i = 0; i < NUM_ITERATIONS; i++)
        {
            if (csrs_nnz && csrd_nnz)
            { // single and double
                if (max == 2)
                {
                    spmv_GPU_SD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_xd, d_y_SD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
                else if (max == 4)
                {
                    spmv_GPU_SD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_xd, d_y_SD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
                else if (max == 8)
                {
                    spmv_GPU_SD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_xd, d_y_SD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
                else if (max == 16)
                {
                    spmv_GPU_SD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_xd, d_y_SD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
                else
                {
                    spmv_GPU_SD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_xd, d_y_SD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
            }
            else if (csrs_nnz && csrd_nnz == 0)
            { // single
                if (max == 2)
                {
                    spmv_GPU_S<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_SD);
                }
                else if (max == 4)
                {
                    spmv_GPU_S<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_SD);
                }
                else if (max == 8)
                {
                    spmv_GPU_S<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_SD);
                }
                else if (max == 16)
                {
                    spmv_GPU_S<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_SD);
                }
                else
                {
                    spmv_GPU_S<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, exponent, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_SD);
                }
            }
            else
            { // double
                if (max == 2)
                {
                    spmv_GPU_D<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_SD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
                else if (max == 4)
                {
                    spmv_GPU_D<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_SD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
                else if (max == 8)
                {
                    spmv_GPU_D<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_SD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
                else if (max == 16)
                {
                    spmv_GPU_D<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_SD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
                else
                {
                    spmv_GPU_D<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_SD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
            }
        }
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&cuda_elapsed_timed, start_event, stop_event); // ms

        SpMV_time_HSD[max] = cuda_elapsed_timed / NUM_ITERATIONS;
    }
    cudaMemcpy(y_SD, d_y_SD, A.nrows * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_csrValS);
    cudaFree(d_csrColIndS);
    cudaFree(d_csrRowPtrS);
    cudaFree(d_csrValD);
    cudaFree(d_csrColIndD);
    cudaFree(d_csrRowPtrD);
    cudaFree(d_xd);
    cudaFree(d_y_SD);
    free(AS.rows);
    free(AD.rows);

    if (csrs_nnz != 0)
    {
        free(AS.cols);
        free(AS.vals);
    }
    if (csrd_nnz != 0)
    {
        free(AD.cols);
        free(AD.vals);
    }
    /****************************************mixed-precision spmv finished*********************************************************************************************************/
    // double precision spmv
    double *d_csrVal;

    cudaMalloc(((void **)(&d_csrVal)), A.nnz * sizeof(double));
    cudaMalloc(((void **)(&d_csrRowPtrD)), (A.nrows + 1) * sizeof(double));
    cudaMalloc(((void **)(&d_csrColIndD)), A.nnz * sizeof(double));
    cudaMalloc(((void **)(&d_xd)), A.ncols * sizeof(double));

    size_t MAX_BLOCKS = 0;
    if (THREADS_PER_VECTORS == 2)
    {
        MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<2>, THREADS_PER_BLOCK, 0);
    }
    else if (THREADS_PER_VECTORS == 4)
    {
        MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<4>, THREADS_PER_BLOCK, 0);
    }
    else if (THREADS_PER_VECTORS == 8)
    {
        MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<8>, THREADS_PER_BLOCK, 0);
    }
    else if (THREADS_PER_VECTORS == 16)
    {
        MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<16>, THREADS_PER_BLOCK, 0);
    }
    else
    {
        MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_D<32>, THREADS_PER_BLOCK, 0);
    }
    const size_t VECTORS_PER_BLOCKS = THREADS_PER_BLOCK / THREADS_PER_VECTORS;
    const size_t NUM_BLOCKSD = min(MAX_BLOCKS, ((A.nrows + (VECTORS_PER_BLOCKS - 1)) / VECTORS_PER_BLOCKS < 1 ? 1 : (A.nrows + (VECTORS_PER_BLOCKS - 1)) / VECTORS_PER_BLOCKS));

    double transfer_time_D = 0;
    double SpMV_time_D = 0;

    cudaEventRecord(start_event, 0);
    // transfer time
    cudaMemcpy(d_csrVal, A.vals, A.nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrRowPtrD, A.rows, (A.nrows + 1) * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIndD, A.cols, A.nnz * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xd, xd, A.ncols * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&cuda_elapsed_timed, start_event, stop_event);
    transfer_time_D = cuda_elapsed_timed / NUM_ITERATIONS;
    // THREADS_PER_VECTORS = 32;
    cudaEventRecord(start_event, 0);
    // begin spmv
    for (i = 0; i < NUM_ITERATIONS; i++)
    {
        if (THREADS_PER_VECTORS == 2)
        {

            spmv_GPU_D<2><<<NUM_BLOCKSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_D, d_csrVal, d_csrColIndD, d_csrRowPtrD);
        }
        else if (THREADS_PER_VECTORS == 4)
        {

            spmv_GPU_D<4><<<NUM_BLOCKSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_D, d_csrVal, d_csrColIndD, d_csrRowPtrD);
        }
        else if (THREADS_PER_VECTORS == 8)
        {

            spmv_GPU_D<8><<<NUM_BLOCKSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_D, d_csrVal, d_csrColIndD, d_csrRowPtrD);
        }
        else if (THREADS_PER_VECTORS == 16)
        {

            spmv_GPU_D<16><<<NUM_BLOCKSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_D, d_csrVal, d_csrColIndD, d_csrRowPtrD);
        }
        else
        {
            spmv_GPU_D<32><<<NUM_BLOCKSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_D, d_csrVal, d_csrColIndD, d_csrRowPtrD);
        }
    }
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&cuda_elapsed_timed, start_event, stop_event);

    SpMV_time_D = cuda_elapsed_timed / NUM_ITERATIONS;

    cudaMemcpy(y_D, d_y_D, A.nrows * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_csrVal);
    cudaFree(d_csrColIndD);
    cudaFree(d_csrRowPtrD);
    cudaFree(d_xd);
    cudaFree(d_y_D);
    /***************************double precision spmv finished******************************************************************************************************/

    /*********************************************************************************************************************************/
    uint k0, k1, k2, k3, k4, k5, k6, k7, k8, ki;
    k0 = k1 = k2 = k3 = k4 = k5 = k6 = k7 = k8 = ki = 0;
    for (i = 0; i < A.nrows; i++)
    {
        double tmp = y_D[i];
        double tmp1 = y_SD[i];
        while ((int)tmp != 0) // normalizes decimal numbers Hari S idea
        {
            tmp1 = tmp1 / 10.0;
            tmp = tmp / 10.0;
        }

        double kor = fabs(tmp1 - tmp);
        if (kor <= 0.0000000005)
        {
            ki++;
        }
        else if (kor <= 0.000000005)
            k8++;
        else if (kor <= 0.00000005)
        {
            k7++;
        }
        else if (kor <= 0.0000005)
            k6++;
        else if (kor <= 0.000005)
            k5++;
        else if (kor <= 0.00005)
            k4++;
        else if (kor <= 0.0005)
        {
            k3++;
        }
        else if (kor <= 0.005)
            k2++;
        else if (kor <= 0.05)
            k1++;
        else
        {
            k0++;
            // printf("0: %d  %f  %f\n", i, y_SD[i], y_D[i]);
        }
    }

    /*********************************************************************************************************************************/
    FILE *fp = fopen(argv[2], "a+");
    fprintf(fp, "%s\t%u\t%u\t%u\t%u\t%u\t%u\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\n",
            argv[1],
            A.nrows, A.ncols, A.nnz,
            csrs_nnz, csrd_nnz,
            THREADS_PER_VECTORS,
            NUM_ITERATIONS, split_time, transfer_time_HSD,
            SpMV_time_HSD[2], SpMV_time_HSD[4], SpMV_time_HSD[8], SpMV_time_HSD[16], SpMV_time_HSD[32],
            transfer_time_D, SpMV_time_D,
            k0, k1, k2, k3, k4, k5, k6, k7, k8, ki);
    fclose(fp);
    /*********************************************************************************************************************************/
    free(y_D);
    free(y_SD);
    free(xd);
    return 0;
}
