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
    for (map<int, uint>::iterator curr = tMap.begin(); curr != tMap.end(); curr++)
        tVector.push_back(make_pair(curr->first, curr->second));

    sort(tVector.begin(), tVector.end(), cmp);
}
int main(int argc, char *argv[])
{
    NUM_ITERATIONS = atoi(argv[3]);
    uint i;
    float split_time = 0;
    // original matrix and three precision submatrices
    SpM A;
    SpM AD;
    SpMS AS;
    SpMH AH;

    // load the matrix in CSR format
    A.readMtx(argv[1]);
    // printf("%u  %u  %u\n", A.nrows, A.ncols, A.nnz);
    // map<int, uint> m;
    // for (i = 0; i < A.nnz; i++)
    // {
    //     int32_t *halfval = (int32_t *)(A.vals + i);
    //     int exponent = ((halfval[1] >> 20) & 0x7ff) - 1023;
    //     m[exponent]++;
    // }
    // vector<pair<int, uint>> tVector;
    // sortMapByValue(m, tVector);
    // FILE *fp1 = fopen(argv[2], "a+");
    // fprintf(fp1, "%s\t%u\t%u\t%u", argv[1],A.nrows, A.ncols, A.nnz);
    // for (int i = 0; i < tVector.size() && i < 20; i++)
    // {
    //     fprintf(fp1, "\t%d", tVector[i].first);
    // }
    // fprintf(fp1, "\n");
    // fprintf(fp1, "%s\t%u\t%u\t%u", argv[1],A.nrows, A.ncols, A.nnz);
    // for (int i = 0; i < tVector.size() && i < 20; i++)
    // {
    //     fprintf(fp1, "\t%d", tVector[i].second);
    // }
    // fprintf(fp1, "\n");
    // fprintf(fp1, "\n");
    // fclose(fp1);
    // return 0;
    // object to record time
    cudaEvent_t start_event, stop_event;
    float cuda_elapsed_timed;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // float and double precision vector x
    float *xs;
    double *xd;

    // result vector y
    float *y_S1; // cusp 单精度csr-coop单精度结果用单精度存
    double *y_D;
    double *y_HSD;

    // nnz in submatrix
    size_t csrh_nnz = 0;
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
    for (uint i = 0; i < NUM_ITERATIONS; i++)
    {
        cudaMemcpy(d_rowPtr, A.rows, (A.nrows + 1) * sizeof(uint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colInd, A.cols, A.nnz * sizeof(uint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vals, A.vals, A.nnz * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&cuda_elapsed_timed, start_event, stop_event);
    transfer_time_HSD += cuda_elapsed_timed / NUM_ITERATIONS;

    // min num of blocks between gpu kernel split_GPU and count_GPU
    uint max_blocks = min(cusp::system::cuda::detail::max_active_blocks(split_GPU, THREADS_PER_BLOCK, 0),
                          cusp::system::cuda::detail::max_active_blocks(count_GPU, THREADS_PER_BLOCK, 0));

    const size_t NUM_BLOCK = min(max_blocks, (A.nrows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // record nnz for every thread in GPU
    uint *d_count_precision;
    uint *h_count_precision = (uint *)malloc(sizeof(uint) * NUM_BLOCK * THREADS_PER_BLOCK * 3);
    cudaMalloc(((void **)(&d_count_precision)), NUM_BLOCK * THREADS_PER_BLOCK * 3 * sizeof(uint));

    const uint num_rows_per_block = (((A.nrows + NUM_BLOCK - 1) / NUM_BLOCK + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * THREADS_PER_BLOCK;
    uint num_rows_per_thread = (num_rows_per_block + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // preprocess——count
    cudaEventRecord(start_event, 0);
    count_GPU<<<NUM_BLOCK, THREADS_PER_BLOCK>>>(A.nrows, num_rows_per_block, d_rowPtr, d_vals, d_count_precision);
    cudaMemcpy(h_count_precision, d_count_precision, NUM_BLOCK * THREADS_PER_BLOCK * 3 * sizeof(uint), cudaMemcpyDeviceToHost);

    // compute where to write for each thread
    for (uint i = 0; i < NUM_BLOCK * THREADS_PER_BLOCK && i * num_rows_per_thread < A.nrows; i++)
    {
        uint tmph = csrh_nnz;
        uint tmps = csrs_nnz;
        uint tmpd = csrd_nnz;
        csrh_nnz += h_count_precision[i * 3];
        csrs_nnz += h_count_precision[i * 3 + 1];
        csrd_nnz += h_count_precision[i * 3 + 2];
        h_count_precision[i * 3] = tmph;
        h_count_precision[i * 3 + 1] = tmps;
        h_count_precision[i * 3 + 2] = tmpd;
    }
    cudaMemcpy(d_count_precision, h_count_precision, NUM_BLOCK * THREADS_PER_BLOCK * 3 * sizeof(uint), cudaMemcpyHostToDevice);

    // memory management
    AH.ncols = A.ncols;
    AH.nrows = A.nrows;
    AH.nnz = csrh_nnz;

    AS.ncols = A.ncols;
    AS.nrows = A.nrows;
    AS.nnz = csrs_nnz;

    AD.ncols = A.ncols;
    AD.nrows = A.nrows;
    AD.nnz = csrd_nnz;

    float *d_xs;
    double *d_xd;

    uint *d_csrRowPtrS;
    uint *d_csrColIndS;
    float *d_csrValS;

    uint *d_csrRowPtrD;
    uint *d_csrColIndD;
    double *d_csrValD;

    half *d_csrValH;
    uint *d_csrRowPtrH;
    uint *d_csrColIndH;

    float *d_y_S1;
    double *d_y_D;
    double *d_y_HSD;

    cudaMalloc(((void **)(&d_xd)), A.ncols * sizeof(double));
    cudaMalloc(((void **)(&d_xs)), A.ncols * sizeof(float));

    cudaMalloc(((void **)(&d_y_HSD)), A.nrows * sizeof(double));
    cudaMalloc(((void **)(&d_y_D)), A.nrows * sizeof(double));
    cudaMalloc(((void **)(&d_y_S1)), A.nrows * sizeof(float));

    cudaMalloc(((void **)(&d_csrValS)), AS.nnz * sizeof(float));
    cudaMalloc(((void **)(&d_csrColIndS)), AS.nnz * sizeof(uint));
    cudaMalloc(((void **)(&d_csrRowPtrS)), (A.nrows + 1) * sizeof(uint));

    cudaMalloc(((void **)(&d_csrValD)), AD.nnz * sizeof(double));
    cudaMalloc(((void **)(&d_csrColIndD)), AD.nnz * sizeof(uint));
    cudaMalloc(((void **)(&d_csrRowPtrD)), (A.nrows + 1) * sizeof(uint));

    cudaMalloc(((void **)(&d_csrValH)), AH.nnz * sizeof(half));
    cudaMalloc(((void **)(&d_csrColIndH)), AH.nnz * sizeof(uint));
    cudaMalloc(((void **)(&d_csrRowPtrH)), (A.nrows + 1) * sizeof(uint));

    // preprocess——split
    split_GPU<<<NUM_BLOCK, THREADS_PER_BLOCK>>>(A.nrows, num_rows_per_block, d_rowPtr, d_colInd, d_vals, d_count_precision, d_csrRowPtrH, d_csrColIndH, d_csrValH, d_csrRowPtrS, d_csrColIndS, d_csrValS, d_csrRowPtrD, d_csrColIndD, d_csrValD);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&cuda_elapsed_timed, start_event, stop_event);

    split_time = cuda_elapsed_timed;

    cudaFree(d_vals);
    cudaFree(d_rowPtr);
    cudaFree(d_colInd);
    // printf("preprocess: %u  %u  %u\n", csrh_nnz, csrs_nnz, csrd_nnz);
    //-------------------------------------------------preprocess  finish----------------------------------------------------------------------------

    // initial vector x
    xs = ((float *)(malloc(sizeof(float) * A.ncols)));
    xd = ((double *)(malloc(sizeof(double) * A.ncols)));

    y_S1 = ((float *)(malloc(sizeof(float) * A.nrows)));
    y_D = ((double *)(malloc(sizeof(double) * A.nrows)));
    y_HSD = ((double *)(malloc(sizeof(double) * A.nrows)));

    srand(time(NULL));
    for (i = 0; i < A.ncols; i++)
    { //[0,1)浮点数
        double tmp = (1 * rand() / (RAND_MAX + 1.0));
        // xs[i] = 1.0;
        // xd[i] = 1.0;
        xs[i] = (float)tmp;
        xd[i] = tmp;
    }

    for (i = 0; i < A.nrows; i++)
    {
        y_S1[i] = 0;
        y_D[i] = 0;
        y_HSD[i] = 0;
    }

    uint avg = (A.nnz + A.nrows - 1) / A.nrows;
    uint sqr = sqrt(avg);
    uint THREADS_PER_VECTORS = cal_vectors(sqr);

    // part of transfertime——vector
    cudaEventRecord(start_event, 0);
    // if (csrh_nnz && csrs_nnz && csrd_nnz == 0)
    // { // half and single
    //     // cudaMemcpy(d_xs, xs, sizeof(float) * A.ncols, cudaMemcpyHostToDevice);
    //     cudaMemcpy(d_xd, xd, sizeof(double) * A.ncols, cudaMemcpyHostToDevice);
    // }
    // else if (csrh_nnz && csrs_nnz == 0 && csrd_nnz)
    // { // half and double
    //     // cudaMemcpy(d_xs, xs, sizeof(float) * A.ncols, cudaMemcpyHostToDevice);
    //     cudaMemcpy(d_xd, xd, sizeof(double) * A.ncols, cudaMemcpyHostToDevice);
    // }
    // else if (csrh_nnz == 0 && csrs_nnz && csrd_nnz)
    // { // single and double
    //     cudaMemcpy(d_xd, xd, sizeof(double) * A.ncols, cudaMemcpyHostToDevice);
    // }
    // else if (csrh_nnz && csrs_nnz && csrd_nnz)
    // { // half , single and double

    //     cudaMemcpy(d_xs, xs, sizeof(float) * A.ncols, cudaMemcpyHostToDevice);
    //     cudaMemcpy(d_xd, xd, sizeof(double) * A.ncols, cudaMemcpyHostToDevice);
    // }
    // else if (csrh_nnz && csrs_nnz == 0 && csrd_nnz == 0)
    // { // half
    //     cudaMemcpy(d_xs, xs, sizeof(float) * A.ncols, cudaMemcpyHostToDevice);
    // }
    // else if (csrh_nnz == 0 && csrs_nnz && csrd_nnz == 0)
    // { // single

    //     cudaMemcpy(d_xd, xd, sizeof(double) * A.ncols, cudaMemcpyHostToDevice);
    // }
    // else
    // { // double
    cudaMemcpy(d_xd, xd, sizeof(double) * A.ncols, cudaMemcpyHostToDevice);
    // }

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&cuda_elapsed_timed, start_event, stop_event);
    transfer_time_HSD += cuda_elapsed_timed;

    int max;
    double SpMV_time_HSD[33];
    size_t MAX_BLOCKS_HSD = 0;

    // test five parameter
    for (max = 2; max <= 32; max = max * 2)
    {

        if (csrd_nnz == 0 && csrs_nnz && csrh_nnz)
        { // single and double
            if (max == 2)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HS<2>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 4)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HS<4>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 8)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HS<8>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 16)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HS<16>, THREADS_PER_BLOCK, 0);
            }
            else
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HS<32>, THREADS_PER_BLOCK, 0);
            }
        }
        else if (csrd_nnz && csrs_nnz == 0 && csrh_nnz)
        { // half and double

            if (max == 2)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HD<2>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 4)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HD<4>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 8)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HD<8>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 16)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HD<16>, THREADS_PER_BLOCK, 0);
            }
            else
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HD<32>, THREADS_PER_BLOCK, 0);
            }
        }
        else if (csrd_nnz && csrs_nnz && csrh_nnz == 0)
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
        else if (csrd_nnz && csrs_nnz && csrh_nnz)
        { // half , single and double
            if (max == 2)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HSD<2>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 4)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HSD<4>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 8)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HSD<8>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 16)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HSD<16>, THREADS_PER_BLOCK, 0);
            }
            else
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_HSD<32>, THREADS_PER_BLOCK, 0);
            }
        }
        else if (csrd_nnz == 0 && csrs_nnz == 0 && csrh_nnz)
        { // half
            if (max == 2)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_Hs<2>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 4)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_Hs<4>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 8)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_Hs<8>, THREADS_PER_BLOCK, 0);
            }
            else if (max == 16)
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_Hs<16>, THREADS_PER_BLOCK, 0);
            }
            else
            {
                MAX_BLOCKS_HSD = cusp::system::cuda::detail::max_active_blocks(spmv_GPU_Hs<32>, THREADS_PER_BLOCK, 0);
            }
            printf("MAX_BLOCKS_HSD=%u \n", MAX_BLOCKS_HSD);
        }
        else if (csrd_nnz == 0 && csrs_nnz && csrh_nnz == 0)
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
        if (csrh_nnz && csrs_nnz && csrd_nnz == 0)
        { // half and single
            if (max == 2)
            {
                spmv_GPU_HS<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_HS<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 4)
            {
                spmv_GPU_HS<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_HS<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 8)
            {
                spmv_GPU_HS<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_HS<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 16)
            {
                spmv_GPU_HS<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_HS<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else
            {
                spmv_GPU_HS<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_HS<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
        }
        else if (csrh_nnz && csrs_nnz == 0 && csrd_nnz)
        { // half and double
            if (max == 2)
            {
                spmv_GPU_HD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_HD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 4)
            {
                spmv_GPU_HD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_HD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 8)
            {
                spmv_GPU_HD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_HD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 16)
            {
                spmv_GPU_HD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_HD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else
            {
                spmv_GPU_HD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_HD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
        }
        else if (csrh_nnz == 0 && csrs_nnz && csrd_nnz)
        { // single and double
            if (max == 2)
            {
                spmv_GPU_SD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
            }
            else if (max == 4)
            {
                spmv_GPU_SD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
            }
            else if (max == 8)
            {
                spmv_GPU_SD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
            }
            else if (max == 16)
            {
                spmv_GPU_SD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
            }
            else
            {
                spmv_GPU_SD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
            }
        }
        else if (csrh_nnz && csrs_nnz && csrd_nnz)
        { // half , single and double
            if (max == 2)
            {
                spmv_GPU_HSD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_HSD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 4)
            {
                spmv_GPU_HSD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_HSD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 8)
            {
                spmv_GPU_HSD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_HSD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 16)
            {
                spmv_GPU_HSD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_HSD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else
            {
                spmv_GPU_HSD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_HSD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
        }
        else if (csrh_nnz && csrs_nnz == 0 && csrd_nnz == 0)
        { // half
            if (max == 2)
            {
                spmv_GPU_Hs<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_Hs<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 4)
            {
                spmv_GPU_Hs<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_Hs<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 8)
            {
                spmv_GPU_Hs<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_Hs<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 16)
            {
                spmv_GPU_Hs<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_Hs<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else
            {
                spmv_GPU_Hs<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                // spmv_GPU_Hs<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
        }
        else if (csrh_nnz == 0 && csrs_nnz && csrd_nnz == 0)
        { // single
            if (max == 2)
            {
                spmv_GPU_S<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_HSD);
            }
            else if (max == 4)
            {
                spmv_GPU_S<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_HSD);
            }
            else if (max == 8)
            {
                spmv_GPU_S<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_HSD);
            }
            else if (max == 16)
            {
                spmv_GPU_S<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_HSD);
            }
            else
            {
                spmv_GPU_S<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_HSD);
            }
        }
        else
        { // double
            if (max == 2)
            {
                spmv_GPU_D<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
            }
            else if (max == 4)
            {
                spmv_GPU_D<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
            }
            else if (max == 8)
            {
                spmv_GPU_D<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
            }
            else if (max == 16)
            {
                spmv_GPU_D<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
            }
            else
            {
                spmv_GPU_D<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
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
            if (csrh_nnz && csrs_nnz && csrd_nnz == 0)
            { // half and single
                if (max == 2)
                {
                    spmv_GPU_HS<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_HS<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 4)
                {
                    spmv_GPU_HS<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_HS<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 8)
                {
                    spmv_GPU_HS<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_HS<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 16)
                {
                    spmv_GPU_HS<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_HS<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else
                {
                    spmv_GPU_HS<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_HS<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
            }
            else if (csrh_nnz && csrs_nnz == 0 && csrd_nnz)
            { // half and double
                if (max == 2)
                {
                    spmv_GPU_HD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_HD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 4)
                {
                    spmv_GPU_HD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_HD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 8)
                {
                    spmv_GPU_HD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_HD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 16)
                {
                    spmv_GPU_HD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_HD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else
                {
                    spmv_GPU_HD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_HD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
            }
            else if (csrh_nnz == 0 && csrs_nnz && csrd_nnz)
            { // single and double
                if (max == 2)
                {
                    spmv_GPU_SD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
                else if (max == 4)
                {
                    spmv_GPU_SD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
                else if (max == 8)
                {
                    spmv_GPU_SD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
                else if (max == 16)
                {
                    spmv_GPU_SD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
                else
                {
                    spmv_GPU_SD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
            }
            else if (csrh_nnz && csrs_nnz && csrd_nnz)
            { // half , single and double
                if (max == 2)
                {
                    spmv_GPU_HSD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_HSD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 4)
                {
                    spmv_GPU_HSD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_HSD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 8)
                {
                    spmv_GPU_HSD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_HSD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 16)
                {
                    spmv_GPU_HSD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_HSD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else
                {
                    spmv_GPU_HSD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_HSD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
            }
            else if (csrh_nnz && csrs_nnz == 0 && csrd_nnz == 0)
            { // half
                if (max == 2)
                {
                    spmv_GPU_Hs<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_Hs<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 4)
                {
                    spmv_GPU_Hs<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_Hs<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 8)
                {
                    spmv_GPU_Hs<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_Hs<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 16)
                {
                    spmv_GPU_Hs<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_Hs<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else
                {
                    spmv_GPU_Hs<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    // spmv_GPU_Hs<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
            }
            else if (csrh_nnz == 0 && csrs_nnz && csrd_nnz == 0)
            { // single
                if (max == 2)
                {
                    spmv_GPU_S<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_HSD);
                }
                else if (max == 4)
                {
                    spmv_GPU_S<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_HSD);
                }
                else if (max == 8)
                {
                    spmv_GPU_S<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_HSD);
                }
                else if (max == 16)
                {
                    spmv_GPU_S<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_HSD);
                }
                else
                {
                    spmv_GPU_S<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_csrValS, d_xd, d_csrColIndS, d_csrRowPtrS, d_y_HSD);
                }
            }
            else
            { // double
                if (max == 2)
                {
                    spmv_GPU_D<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
                else if (max == 4)
                {
                    spmv_GPU_D<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
                else if (max == 8)
                {
                    spmv_GPU_D<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
                else if (max == 16)
                {
                    spmv_GPU_D<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
                else
                {
                    spmv_GPU_D<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD);
                }
            }
        }
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&cuda_elapsed_timed, start_event, stop_event); // ms

        SpMV_time_HSD[max] = cuda_elapsed_timed / NUM_ITERATIONS;
    }
    cudaMemcpy(y_HSD, d_y_HSD, A.nrows * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_csrValS);
    cudaFree(d_csrColIndS);
    cudaFree(d_csrRowPtrS);
    cudaFree(d_csrValH);
    cudaFree(d_csrColIndH);
    cudaFree(d_csrRowPtrH);
    cudaFree(d_csrValD);
    cudaFree(d_csrColIndD);
    cudaFree(d_csrRowPtrD);
    if (d_xd != NULL)
        cudaFree(d_xd);
    cudaFree(d_xs);
    cudaFree(d_y_HSD);
    free(AH.rows);
    free(AS.rows);
    free(AD.rows);

    if (csrh_nnz != 0)
    {
        free(AH.cols);
        free(AH.vals);
    }
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
    for (i = 0; i < NUM_ITERATIONS; i++)
    {
        cudaMemcpy(d_csrVal, A.vals, A.nnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csrRowPtrD, A.rows, (A.nrows + 1) * sizeof(uint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csrColIndD, A.cols, A.nnz * sizeof(uint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_xd, xd, A.ncols * sizeof(double), cudaMemcpyHostToDevice);
    }
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
    double split_time_Mp = 0;
    double transfer_time_Mp = 0;
    double SpMV_time_Mp = 0;
    uint nnzs = 0;
    uint nnzd = 0;
    double *y_Mp = (double *)malloc(sizeof(double) * A.nrows);
    int vectors = 0;
    Mp_SpMV(&A, xs, xd, y_Mp, &split_time_Mp, &transfer_time_Mp, &SpMV_time_Mp, &nnzs, &nnzd, &vectors);

    SpMV_time_Mp = SpMV_time_Mp / NUM_ITERATIONS;

    /*********************************************************************************************************************************/
    uint k0, k1, k2, k3, k4, k5, k6, k7, k8, ki;
    k0 = k1 = k2 = k3 = k4 = k5 = k6 = k7 = k8 = ki = 0;
    for (i = 0; i < A.nrows; i++)
    {
        double tmp = y_D[i];
        double tmp1 = y_HSD[i];
        while ((int)tmp != 0) // normalizes decimal numbers Hari S idea
        {
            tmp1 = tmp1 / 10.0;
            tmp = tmp / 10.0;
        }

        double kor = fabs(tmp1 - tmp);
        if (kor <= 0.0000000005)
            ki++;
        else if (kor <= 0.000000005)
            k8++;
        else if (kor <= 0.00000005)
        {
            k7++;
            printf("7:%d  %f  %f\n", i, y_HSD[i], y_D[i]);
        }
        else if (kor <= 0.0000005)
            k6++;
        else if (kor <= 0.000005)
            k5++;
        else if (kor <= 0.00005)
            k4++;
        else if (kor <= 0.0005)
            k3++;
        else if (kor <= 0.005)
            k2++;
        else if (kor <= 0.05)
            k1++;
        else
        {
            k0++;
            printf("0:%d  %f  %f\n", i, y_HSD[i], y_D[i]);
        }
    }

    uint k0_Mp, k1_Mp, k2_Mp, k3_Mp, k4_Mp, k5_Mp, k6_Mp, k7_Mp, k8_Mp, ki_Mp;
    k0_Mp = k1_Mp = k2_Mp = k3_Mp = k4_Mp = k5_Mp = k6_Mp = k7_Mp = k8_Mp = ki_Mp = 0;
    for (i = 0; i < A.nrows; i++)
    {
        double tmp = y_D[i];
        while ((int)tmp != 0) // normalizes decimal numbers Hari S idea
        {
            y_Mp[i] = y_Mp[i] / 10.0;
            tmp = tmp / 10.0;
        }

        double kor = fabs(y_Mp[i] - tmp);
        if (kor <= 0.0000000005)
            ki_Mp++;
        else if (kor <= 0.000000005)
            k8_Mp++;
        else if (kor <= 0.00000005)
            k7_Mp++;
        else if (kor <= 0.0000005)
            k6_Mp++;
        else if (kor <= 0.000005)
            k5_Mp++;
        else if (kor <= 0.00005)
            k4_Mp++;
        else if (kor <= 0.0005)
            k3_Mp++;
        else if (kor <= 0.005)
            k2_Mp++;
        else if (kor <= 0.05)
            k1_Mp++;
        else
            k0_Mp++;
    }
    /*********************************************************************************************************************************/
    FILE *fp = fopen(argv[2], "a+");
    fprintf(fp, "%s\t%u\t%u\t%u\t%lu\t%lu\t%lu\t%u\t%u\t%u\t%u\t%u\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\n",
            argv[1], A.nrows, A.ncols, A.nnz, csrh_nnz, csrs_nnz,
            csrd_nnz, nnzs, nnzd, THREADS_PER_VECTORS,
            vectors, NUM_ITERATIONS, split_time, transfer_time_HSD, SpMV_time_HSD[2], SpMV_time_HSD[4], SpMV_time_HSD[8], SpMV_time_HSD[16], SpMV_time_HSD[32],
            transfer_time_D, SpMV_time_D, split_time_Mp, transfer_time_Mp, SpMV_time_Mp,
            k0, k1, k2, k3, k4, k5, k6, k7, k8, ki,
            k0_Mp, k1_Mp, k2_Mp, k3_Mp, k4_Mp, k5_Mp, k6_Mp, k7_Mp, k8_Mp, ki_Mp);
    fclose(fp);
    /*********************************************************************************************************************************/
    free(y_D);
    free(y_S1);
    free(y_Mp);
    free(y_HSD);
    free(xs);
    free(xd);
    return 0;
}
