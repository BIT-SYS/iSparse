// /*
// pSpMV源代码
// */

#include <cusp/system/cuda/arch.h>
#include "format.h"
#include "cusp/io/matrix_market.h"
#include "kernel.h"
#include <math.h>
#include <cusparse.h>
int NUM_ITERATIONS;
float test_cusparseSpMV_D(
    uint Am, uint An, uint Annz,
    uint *csrRowPtr, uint *csrColInd, double *csrVal,
    double *xd, double *y, double *transfer_time)
{
    double alpha = 1.0f; //必须用双精度
    double beta = 0.0f;

    //--------------------------------------------------------------------------
    // Device memory management
    uint *dA_csrOffsets, *dA_columns;
    double *dA_values, *d_X, *d_Y;
    cudaMalloc((void **)&dA_csrOffsets, (Am + 1) * sizeof(int));
    cudaMalloc((void **)&dA_columns, Annz * sizeof(int));
    cudaMalloc((void **)&dA_values, Annz * sizeof(double));
    cudaMalloc((void **)&d_X, An * sizeof(double));
    cudaMalloc((void **)&d_Y, Am * sizeof(double));

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t handle = 0;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    cusparseCreate(&handle);
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, Am, An, Annz,
                      dA_csrOffsets, dA_columns, dA_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    // Create dense vector y
    cusparseCreateDnVec(&vecY, Am, d_Y, CUDA_R_64F);
    cudaEvent_t start, stop;
    float time_elapsed = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaMemcpy(dA_csrOffsets, csrRowPtr,
               (Am + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dA_columns, csrColInd, Annz * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dA_values, csrVal,
               Annz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, xd,
               An * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);
    (*transfer_time) = time_elapsed;
    // Create dense vector X
    cusparseCreateDnVec(&vecX, An, d_X, CUDA_R_64F);

    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        // execute SpMV
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                     CUSPARSE_MV_ALG_DEFAULT, dBuffer);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);                       // Waits for an event to complete.Record之前的任务
    cudaEventElapsedTime(&time_elapsed, start, stop); //计算时间差
    cudaMemcpy(y, d_Y, Am * sizeof(double), cudaMemcpyDeviceToHost);
    // destroy matrix/vector descriptors
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);

    // device memory deallocation
    cudaFree(dBuffer);
    cudaFree(dA_csrOffsets);
    cudaFree(dA_columns);
    cudaFree(dA_values);
    return time_elapsed / NUM_ITERATIONS;
}

// int main(int argc, char *argv[])
// {
//     NUM_ITERATIONS = atoi(argv[3]);
//     SpM A;
//     A.readMtx(argv[1]);

//     /**********x向量声明 ：单精度 双精度 半精度**************************/
//     double *xd;

//     /************结果向量 : 双精度 HSD 单精度 SD*******************************/
//     double *y_D;

//     /****************初始化x向量和 结果向量y*******************************************/
//     xd = ((double *)(malloc(sizeof(double) * A.ncols)));

//     y_D = ((double *)(malloc(sizeof(double) * A.nrows)));

//     srand(time(NULL));
//     for (uint i = 0; i < A.ncols; i++)
//     { //[0,1)浮点数
//         double tmp = (1 * rand() / (RAND_MAX + 1.0));
//         xd[i] = tmp;
//     }

//     for (uint i = 0; i < A.nrows; i++)
//     {
//         y_D[i] = 0;
//     }
//     double transfer_time_D = 0;
//     double SpMV_time_D = 0;
//     SpMV_time_D = test_cusparseSpMV_D(A.nrows, A.ncols, A.nnz, A.rows, A.cols, A.vals, xd, y_D, &transfer_time_D);
//     FILE *fp = fopen(argv[2], "a+");
//     // fprintf(fp,"Matrix\tRows\tCols\tnnz\tnnzH\tnnzS\tnnzD\tnnzS_Mp\tnnzD_Mp\t" );
//     fprintf(fp, "%s\t%d\t%d\t%d\t%.6f\t%.6f\n",
//             argv[1], A.nrows, A.ncols, A.nnz, transfer_time_D, SpMV_time_D);
//     fclose(fp);
//     /*********************************************************************************************************************************/
//     free(y_D);
//     free(xd);
//     return 0;
// }
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

void statistics_H(SpMH *csr, uint *min, uint *max, uint *nums, double *variance, double avg)
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
#define HandleError                                               \
    {                                                             \
        cudaError_t cudaError;                                    \
        cudaError = cudaGetLastError();                           \
        if (cudaError != cudaSuccess)                             \
        {                                                         \
            printf("Error: %s\n", cudaGetErrorString(cudaError)); \
            exit(-1);                                             \
        }                                                         \
    }

// API调用错误处理，可以接受CUDA的API函数调用作为参数
#define CHECK_ERROR(error) checkCudaError(error, __FILE__, __LINE__)
//检查CUDA Runtime状态码，可以接受一个指定的提示信息
#define CHECK_STATE(msg) checkCudaState(error, __FILE__, __LINE__)

inline void checkCudaError(cudaError_t error, const char *file, const int line)
{
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA CALL FAILED:" << file << "( " << line << ")- " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}
inline void checkCudaState(const char *msg, const char *file, const int line)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "---" << msg << " Error---" << std::endl;
        std::cerr << file << "( " << line << ")- " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}



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

__global__ void merge_GPU(uint *d_count_precision, uint num_rows_per_thread, uint NUM_BLOCK, uint Am, uint *ret)
{
    size_t csrh_nnz = 0;
    size_t csrs_nnz = 0;
    size_t csrd_nnz = 0;
    for (uint i = 0; i < NUM_BLOCK * THREADS_PER_BLOCK && i * num_rows_per_thread < Am; i++)
    { // compute where to write for each thread
        uint tmph = csrh_nnz;
        uint tmps = csrs_nnz;
        uint tmpd = csrd_nnz;
        csrh_nnz += d_count_precision[i * 3];
        csrs_nnz += d_count_precision[i * 3 + 1];
        csrd_nnz += d_count_precision[i * 3 + 2];
        d_count_precision[i * 3] = tmph;
        d_count_precision[i * 3 + 1] = tmps;
        d_count_precision[i * 3 + 2] = tmpd;
    }
    ret[0] = csrh_nnz;
    ret[1] = csrs_nnz;
    ret[2] = csrd_nnz;
}

int main(int argc, char *argv[])
{

    NUM_ITERATIONS = atoi(argv[3]);
    uint i;
    cudaEvent_t start_event, stop_event; // object to record time
    float cuda_elapsed_timed;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    SpM A;
    A.readMtx(argv[1]); // load the matrix in CSR format
    uint *d_rowPtr;
    uint *d_colInd;
    double *d_vals;
    cudaMalloc((void **)&d_rowPtr, (A.nrows + 1) * sizeof(uint));
    cudaMalloc((void **)&d_colInd, A.nnz * sizeof(uint));
    cudaMalloc((void **)&d_vals, A.nnz * sizeof(double));

    cudaEventRecord(start_event, 0); // part of transfer time —— transfer original matrix
    cudaMemcpy(d_rowPtr, A.rows, (A.nrows + 1) * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd, A.cols, A.nnz * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, A.vals, A.nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&cuda_elapsed_timed, start_event, stop_event);
    double transfer_time_HSD = cuda_elapsed_timed;

    // min num of blocks between gpu kernel split_GPU and count_GPU
    uint max_blocks = min(cusp::system::cuda::detail::max_active_blocks(split_GPU, THREADS_PER_BLOCK, 0),
                          cusp::system::cuda::detail::max_active_blocks(count_GPU, THREADS_PER_BLOCK, 0));
    uint NUM_BLOCK = min(max_blocks, (A.nrows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    // printf("%d\n", NUM_BLOCK);
    // record nnz for every thread in GPU
    uint *d_count_precision;
    cudaMalloc((void **)&d_count_precision, NUM_BLOCK * THREADS_PER_BLOCK * 3 * sizeof(uint));

    uint num_rows_per_block = (A.nrows + NUM_BLOCK - 1) / NUM_BLOCK;
    num_rows_per_block = (num_rows_per_block + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    num_rows_per_block = num_rows_per_block * THREADS_PER_BLOCK;
    // printf("%d\n", num_rows_per_block);
    uint num_rows_per_thread = (num_rows_per_block + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // NUM_BLOCK = (A.nrows+num_rows_per_thread*THREADS_PER_BLOCK-1)/(num_rows_per_thread*THREADS_PER_BLOCK);
    // printf("%d\n",NUM_BLOCK);
    // preprocess——count
    cudaEventRecord(start_event, 0);
    count_GPU<<<NUM_BLOCK, THREADS_PER_BLOCK>>>(A.nrows, num_rows_per_block, d_rowPtr, d_vals, d_count_precision);

    // nnz in submatrix
    size_t csrh_nnz = 0;
    size_t csrs_nnz = 0;
    size_t csrd_nnz = 0;
    // if (A.nrows < +1)
    // {
    uint *h_count_precision = (uint *)malloc(sizeof(uint) * NUM_BLOCK * THREADS_PER_BLOCK * 3);
    cudaMemcpy(h_count_precision, d_count_precision, NUM_BLOCK * THREADS_PER_BLOCK * 3 * sizeof(uint), cudaMemcpyDeviceToHost);
    for (uint i = 0; i < NUM_BLOCK * THREADS_PER_BLOCK && i * num_rows_per_thread < A.nrows; i++)
    { // compute where to write for each thread
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
    free(h_count_precision);
    // }
    // else
    // {
    //     uint *merge_ret = (uint *)malloc(sizeof(uint) * 3);
    //     uint *merge_ret_d;
    //     cudaMalloc((void **)&merge_ret_d, 3 * sizeof(uint));
    //     merge_GPU<<<1, 1>>>(d_count_precision, num_rows_per_thread, NUM_BLOCK, A.nrows, merge_ret_d);
    //     cudaMemcpy(merge_ret, merge_ret_d, 3 * sizeof(uint), cudaMemcpyDeviceToHost);
    //     csrh_nnz = merge_ret[0];
    //     csrs_nnz = merge_ret[1];
    //     csrd_nnz = merge_ret[2];
    //     free(merge_ret);
    //     cudaFree(merge_ret_d);
    // }
    // printf("%u  %u  %u  %u\n", A.nnz, csrh_nnz, csrs_nnz, csrd_nnz);
    SpM AD;
    SpMS AS;
    SpMH AH;
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

    uint *d_csrRowPtrS;
    uint *d_csrColIndS;
    float *d_csrValS;

    uint *d_csrRowPtrD;
    uint *d_csrColIndD;
    double *d_csrValD;

    half *d_csrValH;
    uint *d_csrRowPtrH;
    uint *d_csrColIndH;

    cudaMalloc((void **)&d_csrValS, AS.nnz * sizeof(float));
    cudaMalloc((void **)&d_csrColIndS, AS.nnz * sizeof(uint));
    cudaMalloc((void **)&d_csrRowPtrS, (A.nrows + 1) * sizeof(uint));

    cudaMalloc((void **)&d_csrValD, AD.nnz * sizeof(double));
    cudaMalloc((void **)&d_csrColIndD, AD.nnz * sizeof(uint));
    cudaMalloc((void **)&d_csrRowPtrD, (A.nrows + 1) * sizeof(uint));

    cudaMalloc((void **)&d_csrValH, AH.nnz * sizeof(half));
    cudaMalloc((void **)&d_csrColIndH, AH.nnz * sizeof(uint));
    cudaMalloc((void **)&d_csrRowPtrH, (A.nrows + 1) * sizeof(uint));

    // preprocess——split

    split_GPU<<<NUM_BLOCK, THREADS_PER_BLOCK>>>(A.nrows, num_rows_per_block, d_rowPtr, d_colInd, d_vals, d_count_precision, d_csrRowPtrH, d_csrColIndH, d_csrValH, d_csrRowPtrS, d_csrColIndS, d_csrValS, d_csrRowPtrD, d_csrColIndD, d_csrValD);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&cuda_elapsed_timed, start_event, stop_event);

    float split_time = cuda_elapsed_timed;
    cudaFree(d_vals);
    cudaFree(d_rowPtr);
    cudaFree(d_colInd);
    cudaFree(d_count_precision);

    // AH.rows = (uint *)malloc(sizeof(uint) * (AH.nrows + 1));
    // AS.rows = (uint *)malloc(sizeof(uint) * (AS.nrows + 1));
    // AD.rows = (uint *)malloc(sizeof(uint) * (AD.nrows + 1));
    // cudaMemcpy(AH.rows, d_csrRowPtrH, sizeof(uint) * (AH.nrows + 1), cudaMemcpyDeviceToHost);
    // cudaMemcpy(AS.rows, d_csrRowPtrS, sizeof(uint) * (AS.nrows + 1), cudaMemcpyDeviceToHost);
    // cudaMemcpy(AD.rows, d_csrRowPtrD, sizeof(uint) * (AD.nrows + 1), cudaMemcpyDeviceToHost);
    // double avgs = AS.nnz / AS.nrows;
    // avgs = max(avgs, 1);
    // double avgd = max(AD.nnz / AD.nrows, 1);
    // double avgh = max(AH.nnz / AH.nrows, 1);

    // uint minh = 0;
    // uint mins = 0;
    // uint mind = 0;
    // uint maxh = 0;
    // uint maxs = 0;
    // uint maxd = 0;
    // uint numh = 0;
    // uint nums = 0;
    // uint numd = 0;
    // double varh = 0;
    // double vars = 0;
    // double vard = 0;
    // if (csrh_nnz)
    //     statistics_H(&AH, &minh, &maxh, &numh, &varh, (double)(avgh));
    // if (csrs_nnz)
    //     statistics_S(&AS, &mins, &maxs, &nums, &vars, (double)(avgs));
    // if (csrd_nnz)
    //     statistics_D(&AD, &mind, &maxd, &numd, &vard, (double)(avgd));
    // FILE *fp1 = fopen(argv[2], "a+");
    // fprintf(fp1, "%s\t%u\t%u\t%u\t%u\t%u\t%u\t%f\t%u\t%u\t%u\t%f\t%u\t%u\t%u\t%f\t%u\t%u\t%u\t%f\t%f\t%f\n",
    //         argv[1],
    //         A.nrows, A.ncols, A.nnz, csrh_nnz, csrs_nnz, csrd_nnz,
    //         avgh, minh, maxh, numh,
    //         avgs, mins, maxs, nums,
    //         avgd, mind, maxd, numd,
    //         varh / (double)A.nrows, vars / (double)A.nrows, vard / (double)A.nrows);
    // fclose(fp1);
    // printf("%u  %u  %u  %u\n", A.nnz, AH.nnz, AS.nnz, AD.nnz);
    //-------------------------------------------------preprocess  finish----------------------------------------------------------------------------
    float *d_xs;
    double *d_xd;
    double *d_y_HSD;
    cudaMalloc((void **)&d_xd, A.ncols * sizeof(double));
    cudaMalloc((void **)&d_xs, A.ncols * sizeof(float));
    cudaMalloc((void **)&d_y_HSD, A.nrows * sizeof(double));

    float *xs;
    double *xd;
    xs = (float *)malloc(sizeof(float) * A.ncols);
    xd = (double *)malloc(sizeof(double) * A.ncols);

    double *y_D;
    double *y_HSD;
    y_D = (double *)malloc(sizeof(double) * A.nrows);
    y_HSD = (double *)malloc(sizeof(double) * A.nrows);

    srand(time(NULL));
    for (i = 0; i < A.ncols; i++)
    { //[0,1)浮点数
        double tmp = (1 * rand() / (RAND_MAX + 1.0));
        // xs[i] = 1.0;
        // xd[i] = 1.0;
        xs[i] = (float)tmp;
        xd[i] = tmp;
    }

    // part of transfertime——vector
    cudaEventRecord(start_event, 0);
    if (csrh_nnz && csrs_nnz && csrd_nnz == 0) // half and single
    {
        cudaMemcpy(d_xs, xs, sizeof(float) * A.ncols, cudaMemcpyHostToDevice);
        cudaMemcpy(d_xd, xd, sizeof(double) * A.ncols, cudaMemcpyHostToDevice);
    }
    else if (csrh_nnz && csrs_nnz == 0 && csrd_nnz) // half and double
    {
        cudaMemcpy(d_xs, xs, sizeof(float) * A.ncols, cudaMemcpyHostToDevice);
        cudaMemcpy(d_xd, xd, sizeof(double) * A.ncols, cudaMemcpyHostToDevice);
    }
    else if (csrh_nnz == 0 && csrs_nnz && csrd_nnz) // single and double
    {
        cudaMemcpy(d_xd, xd, sizeof(double) * A.ncols, cudaMemcpyHostToDevice);
    }
    else if (csrh_nnz && csrs_nnz && csrd_nnz) // half , single and double
    {
        cudaMemcpy(d_xs, xs, sizeof(float) * A.ncols, cudaMemcpyHostToDevice);
        cudaMemcpy(d_xd, xd, sizeof(double) * A.ncols, cudaMemcpyHostToDevice);
    }
    else if (csrh_nnz && csrs_nnz == 0 && csrd_nnz == 0) // half
    {
        cudaMemcpy(d_xs, xs, sizeof(float) * A.ncols, cudaMemcpyHostToDevice);
    }
    else if (csrh_nnz == 0 && csrs_nnz && csrd_nnz == 0) // single
    {
        cudaMemcpy(d_xd, xd, sizeof(double) * A.ncols, cudaMemcpyHostToDevice);
    }
    else // double
    {
        cudaMemcpy(d_xd, xd, sizeof(double) * A.ncols, cudaMemcpyHostToDevice);
    }
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&cuda_elapsed_timed, start_event, stop_event);
    transfer_time_HSD += cuda_elapsed_timed;
    HandleError;
    // printf("transfer ok!");

    uint max;
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
        // printf("%u\n", NUM_BLOCKS_HSD);
        float time;
        cudaEventRecord(start_event, 0);
        // spmv start
        if (csrh_nnz && csrs_nnz && csrd_nnz == 0)
        { // half and single
            if (max == 2)
            {
                // spmv_GPU_HS<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                spmv_GPU_HS<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 4)
            {
                // spmv_GPU_HS<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                spmv_GPU_HS<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 8)
            {
                // spmv_GPU_HS<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                spmv_GPU_HS<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 16)
            {
                // spmv_GPU_HS<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                spmv_GPU_HS<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else
            {
                // spmv_GPU_HS<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                spmv_GPU_HS<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
        }
        else if (csrh_nnz && csrs_nnz == 0 && csrd_nnz)
        { // half and double
            if (max == 2)
            {
                // spmv_GPU_HD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                spmv_GPU_HD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 4)
            {
                // spmv_GPU_HD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                spmv_GPU_HD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 8)
            {
                // spmv_GPU_HD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                spmv_GPU_HD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 16)
            {
                // spmv_GPU_HD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                spmv_GPU_HD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else
            {
                // spmv_GPU_HD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                spmv_GPU_HD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
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
                // spmv_GPU_HSD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                spmv_GPU_HSD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 4)
            {
                // spmv_GPU_HSD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                spmv_GPU_HSD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 8)
            {
                // spmv_GPU_HSD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                spmv_GPU_HSD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 16)
            {
                // spmv_GPU_HSD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                spmv_GPU_HSD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else
            {
                // spmv_GPU_HSD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                spmv_GPU_HSD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
        }
        else if (csrh_nnz && csrs_nnz == 0 && csrd_nnz == 0)
        { // half
            if (max == 2)
            {
                spmv_GPU_Hs<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 4)
            {
                spmv_GPU_Hs<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 8)
            {
                spmv_GPU_Hs<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else if (max == 16)
            {
                spmv_GPU_Hs<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
            }
            else
            {
                spmv_GPU_Hs<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
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
        // printf("%d  %d\n", max, NUM_ITERATIONS);
        cudaEventRecord(start_event, 0);
        for (i = 0; i < NUM_ITERATIONS; i++)
        {
            if (csrh_nnz && csrs_nnz && csrd_nnz == 0)
            { // half and single
                if (max == 2)
                {
                    // spmv_GPU_HS<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    spmv_GPU_HS<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 4)
                {
                    // spmv_GPU_HS<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    spmv_GPU_HS<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 8)
                {
                    // spmv_GPU_HS<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    spmv_GPU_HS<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 16)
                {
                    // spmv_GPU_HS<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    spmv_GPU_HS<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else
                {
                    // spmv_GPU_HS<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    spmv_GPU_HS<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
            }
            else if (csrh_nnz && csrs_nnz == 0 && csrd_nnz)
            { // half and double
                if (max == 2)
                {
                    // spmv_GPU_HD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    spmv_GPU_HD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 4)
                {
                    // spmv_GPU_HD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    spmv_GPU_HD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 8)
                {
                    // spmv_GPU_HD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    spmv_GPU_HD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 16)
                {
                    // spmv_GPU_HD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    spmv_GPU_HD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else
                {
                    // spmv_GPU_HD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    spmv_GPU_HD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
            }
            else if (csrh_nnz == 0 && csrs_nnz && csrd_nnz)
            { // single and half
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
                    // spmv_GPU_HSD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    spmv_GPU_HSD<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 4)
                {
                    // spmv_GPU_HSD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    spmv_GPU_HSD<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 8)
                {
                    // spmv_GPU_HSD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    spmv_GPU_HSD<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 16)
                {
                    // spmv_GPU_HSD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    spmv_GPU_HSD<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else
                {
                    // spmv_GPU_HSD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                    spmv_GPU_HSD<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_xd, d_y_HSD, d_csrValS, d_csrColIndS, d_csrRowPtrS, d_csrValD, d_csrColIndD, d_csrRowPtrD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
            }
            else if (csrh_nnz && csrs_nnz == 0 && csrd_nnz == 0)
            { // half
                if (max == 2)
                {
                    spmv_GPU_Hs<2><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 4)
                {
                    spmv_GPU_Hs<4><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 8)
                {
                    spmv_GPU_Hs<8><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else if (max == 16)
                {
                    spmv_GPU_Hs<16><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
                }
                else
                {
                    spmv_GPU_Hs<32><<<NUM_BLOCKS_HSD, THREADS_PER_BLOCK, 0>>>(A.nrows, d_xs, d_y_HSD, d_csrValH, d_csrColIndH, d_csrRowPtrH);
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

    if (csrs_nnz)
        cudaFree(d_csrValS);
    cudaFree(d_csrColIndS);
    cudaFree(d_csrRowPtrS);

    if (csrh_nnz)
        cudaFree(d_csrValH);
    cudaFree(d_csrColIndH);
    cudaFree(d_csrRowPtrH);

    if (csrd_nnz)
        cudaFree(d_csrValD);
    cudaFree(d_csrColIndD);
    cudaFree(d_csrRowPtrD);

    if (d_xd != NULL)
        cudaFree(d_xd);
    cudaFree(d_xs);
    cudaFree(d_y_HSD);

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
    double *d_y_D;
    cudaMalloc((void **)&d_y_D, A.nrows * sizeof(double));
    cudaMalloc((void **)&d_csrVal, A.nnz * sizeof(double));
    cudaMalloc((void **)&d_csrRowPtrD, (A.nrows + 1) * sizeof(double));
    cudaMalloc((void **)&d_csrColIndD, A.nnz * sizeof(double));
    cudaMalloc((void **)&d_xd, A.ncols * sizeof(double));

    uint avg = (A.nnz + A.nrows - 1) / A.nrows;
    uint sqr = sqrt(avg);
    uint THREADS_PER_VECTORS = cal_vectors(sqr);

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
    /*********************************************MpSPMV finished************************************************************************************/

    /*********************************************************************************************************************************/
    uint k0, k1, k2, k3, k4, k5, k6, k7, k8, ki;
    k0 = k1 = k2 = k3 = k4 = k5 = k6 = k7 = k8 = ki = 0;
    for (i = 0; i < A.nrows; i++)
    {
        double tmp = y_D[i];
        while ((uint)tmp != 0) // normalizes decimal numbers Hari S idea
        {
            y_HSD[i] = y_HSD[i] / 10.0;
            tmp = tmp / 10.0;
        }

        double kor = fabs(y_HSD[i] - tmp);
        if (kor <= 0.0000000005)
            ki++;
        else if (kor <= 0.000000005)
            k8++;
        else if (kor <= 0.00000005)
            k7++;
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
            k0++;
    }

    uint k0_Mp, k1_Mp, k2_Mp, k3_Mp, k4_Mp, k5_Mp, k6_Mp, k7_Mp, k8_Mp, ki_Mp;
    k0_Mp = k1_Mp = k2_Mp = k3_Mp = k4_Mp = k5_Mp = k6_Mp = k7_Mp = k8_Mp = ki_Mp = 0;
    for (i = 0; i < A.nrows; i++)
    {
        double tmp = y_D[i];
        while ((uint)tmp != 0) // normalizes decimal numbers Hari S idea
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
    fprintf(fp, "%s\t%u\t%u\t%u\t%lu\t%lu\t%lu\t%u\t%u\t%u\t%d\t%u\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\n",
            argv[1], A.nrows, A.ncols, A.nnz, csrh_nnz, csrs_nnz,
            csrd_nnz, nnzs, nnzd, THREADS_PER_VECTORS,
            vectors, NUM_ITERATIONS, split_time, transfer_time_HSD, SpMV_time_HSD[2], SpMV_time_HSD[4], SpMV_time_HSD[8], SpMV_time_HSD[16], SpMV_time_HSD[32],
            transfer_time_D, SpMV_time_D, split_time_Mp, transfer_time_Mp, SpMV_time_Mp,
            k0, k1, k2, k3, k4, k5, k6, k7, k8, ki,
            k0_Mp, k1_Mp, k2_Mp, k3_Mp, k4_Mp, k5_Mp, k6_Mp, k7_Mp, k8_Mp, ki_Mp);
    fclose(fp);
    /*********************************************************************************************************************************/
    free(y_D);
    free(y_Mp);
    free(y_HSD);
    free(xs);
    free(xd);

    return 0;
}
