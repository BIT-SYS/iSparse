#include <spamg.h>

// old upper
#include <csr_multiply.h>
#include <csr_multiply_sm35.h>
#include <csr_multiply_sm70.h>
#include <csr_multiply_sm70_upper.h>

// opsparse
#include "CSR_opsparse.h"
#include "Meta.h"

#include <fstream>
#include <cuda_profiler_api.h>
#include <cub/cub.cuh>
#include "kernel_wrapper.cuh"

// spECK
#include "Executor.h"
#include "Multiply.h"
#include <iomanip>
#include "Config.h"
#include "Timings.h"
#include "spECKConfig.h"

// Global includes
#include <bitset>
#include <memory>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Local includes
#include "GPU/spECKKernels.h"
#include "GPU/consistent_gpu_memory.h"
#include "CUDATools/stream.h"
#include "meta_utils.h"
#include "GPU/spECK_HashSpGEMM.cuh"
#include "GPU/spECK_HashLoadBalancer.cuh"
#include "GPU/HelperFunctions.cuh"
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include "common.h"
#include "WorkDistribution.h"
#include "HashMap.cuh"
#include "memory.h"
#include "time.h"

using IndexType = uint32_t;

void startTimerVar(cudaEvent_t &start, CUstream stream = 0)
{
    HANDLE_ERROR(cudaEventRecord(start, stream));
    HANDLE_ERROR(cudaEventSynchronize(start));
}

float recordTimerVar(cudaEvent_t &start, cudaEvent_t &end, CUstream stream = 0)
{
    float time;
    HANDLE_ERROR(cudaEventRecord(end, stream));
    HANDLE_ERROR(cudaEventSynchronize(end));
    HANDLE_ERROR(cudaEventElapsedTime(&time, start, end));
    return time;
}

// 默认实现
namespace amgx
{
    template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
    template<typename TConfig>
    void SpAMG<V, M, I>::spmv_scaler(Matrix<TConfig> &A, Vector<TConfig> &B, Vector<TConfig> &C)
    {
        printf("trivial: spmv_scaler\n");
    }

    template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
    template<typename TConfig>
    void SpAMG<V, M, I>::spmv_cusparse(Matrix<TConfig> &A, Vector<TConfig> &B, Vector<TConfig> &C)
    {
        printf("trivial: spmv_cusparse\n");
    }

    template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
    template<typename TConfig>
    void SpAMG<V, M, I>::spmv_adaptive(Matrix<TConfig> &A, Vector<TConfig> &B, Vector<TConfig> &C)
    {
        printf("trivial: spmv_adaptive\n");
    }

    template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
    template<typename TConfig>
    void SpAMG<V, M, I>::spmv(Matrix<TConfig> &A, Vector<TConfig> &B, Vector<TConfig> &C)
    {
        printf("trivial: spmv\n");
    }

    template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
    void SpAMG<V, M, I>::spgemm_opsparse(Matrix_d &A, Matrix_d &B, Matrix_d &C)
    {
        printf("trivial: spgemm_opsparse\n");
    }

    template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
    void SpAMG<V, M, I>::spgemm_speck(Matrix_d &A, Matrix_d &B, Matrix_d &C)
    {
        printf("trivial: spgemm_speck\n");
    }

    template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
    void SpAMG<V, M, I>::spgemm_upper(Matrix_d &A, Matrix_d &B, Matrix_d &C)
    {
        printf("trivial: spgemm_upper\n");
    }

    template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
    void SpAMG<V, M, I>::spgemm_old(Matrix_d &A, Matrix_d &B, Matrix_d &C)
    {
        printf("trivial: spgemm_old\n");
    }
    template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
    void SpAMG<V, M, I>::spgemm( Matrix_d &A,  Matrix_d &B, Matrix_d &C)
    {
        printf("trivial: spgemm\n");
    }

    template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
    int SpAMG<V, M, I>::level2_AP = -1;
    template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
    int SpAMG<V, M, I>::level2_RAP = -1;
    template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
    double SpAMG<V, M, I>::spgemm_time = 0;
    template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
    double SpAMG<V, M, I>::spmv_time = 0;
    template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
    timeval SpAMG<V, M, I>::tv0 = {0, 0};
    template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
    timeval SpAMG<V, M, I>::tv1 = {0, 0};

}

// 特化实现
namespace amgx
{
    __global__ void spmv_scaler_kernel(
            unsigned int n_rows,
            int *col_ids,
            int *row_ptr,
            const double *data,
            const double *x,
            double *y)
    {
        unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < n_rows)
        {
            const int row_start = row_ptr[row];
            const int row_end = row_ptr[row + 1];

            double sum = 0;
            for (unsigned int element = row_start; element < row_end; element++)
            sum += data[element] * x[col_ids[element]];
            y[row] = sum;
        }
    }
    
    __device__ unsigned int prev_power_of_2(unsigned int n)
    {
        while (n & n - 1)
            n = n & n - 1;
        return n;
    }

    #define FULL_WARP_MASK 0xFFFFFFFF
    __device__ double warp_reduce(double val)
    {
        /**
         *  For a thread at lane X in the warp, __shfl_down_sync(FULL_MASK, val, offset) gets
         *  the value of the val variable from the thread at lane X+offset of the same warp.
         *  The data exchange is performed between registers, and more efficient than going
         *  through shared memory, which requires a load, a store and an extra register to
         *  hold the address.
         */
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            val += __shfl_down_sync(FULL_WARP_MASK, val, offset);

        return val;
    }

    __global__ void csr_adaptive_spmv_kernel(
        const unsigned int n_rows,
        int *col_ids,
        int *row_ptr,
        int *row_blocks,
        const double *data,
        const double *x,
        double *y)
    {
        const unsigned int block_row_begin = row_blocks[blockIdx.x];
        const unsigned int block_row_end = row_blocks[blockIdx.x + 1];
        const unsigned int nnz = row_ptr[block_row_end] - row_ptr[block_row_begin];

        __shared__ double cache[64u];

        if (block_row_end - block_row_begin > 1)
        {
            /// CSR-Stream case
            const unsigned int i = threadIdx.x;
            const unsigned int block_data_begin = row_ptr[block_row_begin];
            const unsigned int thread_data_begin = block_data_begin + i;

            if (i < nnz)
            cache[i] = data[thread_data_begin] * x[col_ids[thread_data_begin]];
            __syncthreads();

            const unsigned int threads_for_reduction = prev_power_of_2(blockDim.x / (block_row_end - block_row_begin));

            if (threads_for_reduction > 1)
            {
            /// Reduce all non zeroes of row by multiple thread
            const unsigned int thread_in_block = i % threads_for_reduction;
            const unsigned int local_row = block_row_begin + i / threads_for_reduction;

            double dot = 0.0;

            if (local_row < block_row_end)
            {
                const unsigned int local_first_element = row_ptr[local_row] - row_ptr[block_row_begin];
                const unsigned int local_last_element = row_ptr[local_row + 1] - row_ptr[block_row_begin];

                for (unsigned int local_element = local_first_element + thread_in_block;
                    local_element < local_last_element;
                    local_element += threads_for_reduction)
                {
                dot += cache[local_element];
                }
            }
            __syncthreads();
            cache[i] = dot;

            /// Now each row has threads_for_reduction values in cache
            for (int j = threads_for_reduction / 2; j > 0; j /= 2)
            {
                /// Reduce for each row
                __syncthreads();

                const bool use_result = thread_in_block < j && i + j < 64u;

                if (use_result)
                dot += cache[i + j];
                __syncthreads();

                if (use_result)
                cache[i] = dot;
            }

            if (thread_in_block == 0 && local_row < block_row_end)
                y[local_row] = dot;
            }
            else
            {
            /// Reduce all non zeroes of row by single thread
            unsigned int local_row = block_row_begin + i;
            while (local_row < block_row_end)
            {
                double dot = 0.0;

                for (unsigned int j = row_ptr[local_row] - block_data_begin;
                    j < row_ptr[local_row + 1] - block_data_begin;
                    j++)
                {
                dot += cache[j];
                }

                y[local_row] = dot;
                local_row += 64u;
            }
            }
        }
        else
        {
            const unsigned int row = block_row_begin;
            const unsigned int warp_id = threadIdx.x / 32;
            const unsigned int lane = threadIdx.x % 32;

            double dot = 0;

            if (nnz <= 64 || 64u <= 32)
            {
            /// CSR-Vector case
            if (row < n_rows)
            {
                const unsigned int row_start = row_ptr[row];
                const unsigned int row_end = row_ptr[row + 1];

                for (unsigned int element = row_start + lane; element < row_end; element += 32)
                dot += data[element] * x[col_ids[element]];
            }

            dot = warp_reduce(dot);

            if (lane == 0 && warp_id == 0 && row < n_rows)
            {
                y[row] = dot;
            }
            }
            else
            {
            /// CSR-VectorL case
            if (row < n_rows)
            {
                const unsigned int row_start = row_ptr[row];
                const unsigned int row_end = row_ptr[row + 1];

                for (unsigned int element = row_start + threadIdx.x; element < row_end; element += blockDim.x)
                dot += data[element] * x[col_ids[element]];
            }

            dot = warp_reduce(dot);

            if (lane == 0)
                cache[warp_id] = dot;
            __syncthreads();

            if (warp_id == 0)
            {
                dot = 0.0;

                for (unsigned int element = lane; element < blockDim.x / 32; element += 32)
                dot += cache[element];

                dot = warp_reduce(dot);

                if (lane == 0 && row < n_rows)
                {
                y[row] = dot;
                }
            }
            }
        }
    }


    //      cuSparse
    void h_spmv_cusparse(
        int MA,
        int NA,
        int nnzA,
        int *dA_csrOffsets, 
        int *dA_columns,
        double *dA_values, 
        double *d_x, 
        double *d_y)
    {
        double alpha = 1.0f;
        double beta = 0.0f;
        cudaDataType computeType = CUDA_R_64F;

        
        // cudaMalloc((void **)&dA_csrOffsets, (MA + 1) * sizeof(int));
        // cudaMalloc((void **)&dA_columns, nnzA * sizeof(int));
        // cudaMalloc((void **)&dA_values, nnzA * sizeof(double));
        // cudaMalloc((void **)&d_x, NA * sizeof(double));
        // cudaMalloc((void **)&d_y, MA * sizeof(double));

        //--------------------------------------------------------------------------
        // CUSPARSE APIs
        cusparseHandle_t handle = 0;
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;
        void *dBuffer = NULL;
        size_t bufferSize = 0;

        (cusparseCreate(&handle));
        // Create sparse matrix A in CSR format
        cusparseCreateCsr(&matA, MA, NA, nnzA,
                            dA_csrOffsets, dA_columns, dA_values,
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO, computeType);
        // Create dense vector y
        cusparseCreateDnVec(&vecY, MA, d_y, computeType);

        // transfer data
        // cudaMemcpy(dA_csrOffsets, csrRowIndexHostPtrA, (MA + 1) * sizeof(int), cudaMemcpyHostToDevice);
        // cudaMemcpy(dA_columns, csrColIndexHostPtrA, nnzA * sizeof(int), cudaMemcpyHostToDevice);
        // cudaMemcpy(dA_values, csrValHostPtrA, nnzA * sizeof(double), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_x, xHostPtr, NA * sizeof(double), cudaMemcpyHostToDevice);

        // Create dense vector X
        cusparseCreateDnVec(&vecX, NA, d_x, computeType);
        cudaMemset(d_y, 0.0, sizeof(double) * MA);

        // check SpMV
        (cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha, matA, vecX, &beta, vecY, computeType, CUSPARSE_MV_ALG_DEFAULT, &bufferSize));
        (cudaMalloc((void **)&dBuffer, bufferSize));
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY, computeType,
                    CUSPARSE_MV_ALG_DEFAULT, dBuffer);
        // cudaMemcpy(yHostPtr, d_y, MA * sizeof(double), cudaMemcpyDeviceToHost);

        // warmup SpMV
        // for (int i = 0; i < 10; i++)
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY, computeType,
                    CUSPARSE_MV_ALG_DEFAULT, dBuffer);

        // time SpMV
        // float time_elapsed = 0;
        // cudaEvent_t start, stop;
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);

        // cudaEventRecord(start);
        // for (int i = 0; i < N; i++)
        // {
        //     cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                 &alpha, matA, vecX, &beta, vecY, computeType,
        //                 CUSPARSE_MV_ALG_DEFAULT, dBuffer);
        // }
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);                       // Waits for an event to complete.Record之前的任务
        // cudaEventElapsedTime(&time_elapsed, start, stop); //计算时间差

        // cudaMemcpy(yHostPtr, d_y, MA * sizeof(double), cudaMemcpyDeviceToHost);

        // destroy matrix/vector descriptors
        cusparseDestroySpMat(matA);
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        cusparseDestroy(handle);
        // device memory deallocation
        cudaFree(dBuffer);
        // cudaFree(dA_csrOffsets);
        // cudaFree(dA_columns);
        // cudaFree(dA_values);
        // spmv_time = time_elapsed / (float)N;

        // return spmv_time;
    }
    //      scaler
    void h_spmv_scaler(
        int MA,
        int NA,
        int nnzA,
        unsigned int n_rows,
        int *row_ptr,
        int *col_ids,
        double *data,
        double *x,
        double *y)
    {
        int block_size{};
        int min_grid_size{};

        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, spmv_scaler_kernel, 0, 0);
        int grid_size = (MA + block_size - 1) / block_size;
        spmv_scaler_kernel<<<grid_size, block_size>>>(n_rows, col_ids, row_ptr, data, x, y);
    }
    //      adaptive
    void h_spmv_adaptive(
        int MA,
        int NA,
        int nnzA,
        int *dA_csrOffsets, 
        int *dA_columns,
        double *dA_values, 
        double *d_x, 
        double *d_y,
        int blocks_count,
        int *d_row_blocks
    )
    {
        // const int blocks_count = fill_row_blocks(false, MA, csrRowIndexHostPtrA, nullptr);
        // std::unique_ptr<int[]> row_blocks(new int[blocks_count + 1]);
        // fill_row_blocks(true, MA, csrRowIndexHostPtrA, row_blocks.get());

        // int *d_row_blocks{};
        // cudaMalloc(&d_row_blocks, (blocks_count + 1) * sizeof(int));
        // cudaMemcpy(d_row_blocks, row_blocks.get(), sizeof(int) * (blocks_count + 1), cudaMemcpyHostToDevice);


        dim3 block_size = dim3(64u);
        dim3 grid_size{};
        grid_size.x = blocks_count;
        // printf("adatptive %d %d\n", grid_size.x, block_size.x);
        csr_adaptive_spmv_kernel<<<grid_size, block_size>>>(MA, dA_columns, dA_csrOffsets, d_row_blocks, dA_values, d_x, d_y);
    }


    template<> template<>
    void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_scaler<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &C)
    {
        // printf("spmv_scaler\n");

        h_spmv_scaler(
            (int) A.get_num_rows(),
            (int) A.get_num_cols(),
            (int) A.get_num_nz(),
            (unsigned int) A.get_num_rows(),
            (int *) A.row_offsets.raw(),
            (int *) A.col_indices.raw(),
            (double *) A.values.raw(),
            (double *) B.raw(),
            (double *) C.raw());

        // printf("A %d %d %d\n", 1,2,3);
    }
    template<> template<>
    void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_cusparse<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &C)
    {
        // printf("spmv_cusparse\n");
        // printf("A %d %d %d\n", 1,2,3);

        // h_spmv_cusparse(
        //     (int) A.get_num_rows(),
        //     (int) A.get_num_cols(),
        //     (int) A.get_num_nz(),

        //     (int *) A.row_offsets.raw(),
        //     (int *) A.col_indices.raw(),
        //     (double *) A.values.raw(),

        //     (double *) B.raw(),
        //     (double *) C.raw());
        
        typedef typename TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::VecPrec ValueTypeB;
        Cusparse::bsrmv<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>>(types::util<ValueTypeB>::get_one(), A, B, types::util<ValueTypeB>::get_zero(), C, A.getViewExterior());
                
    }
    template<> template<>
    void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_adaptive<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &C)
    {
        // printf("spmv_adaptive\n");
        // printf("A %d %d %d\n", 1,2,3);
        if(A.optalgo.d_row_blocks == nullptr) A.optalgo.calc_d_row_blocks((int) A.get_num_rows(), (int *) A.h_row_offsets);
        // printf("adaptive A.optalgo.block_count = %d\n", A.optalgo.block_count);
        h_spmv_adaptive(
            (int) A.get_num_rows(),
            (int) A.get_num_cols(),
            (int) A.get_num_nz(),

            (int *) A.row_offsets.raw(),
            (int *) A.col_indices.raw(),
            (double *) A.values.raw(),

            (double *) B.raw(),
            (double *) C.raw(),

            A.optalgo.block_count,
            A.optalgo.d_row_blocks   
        );
    }
/*
    template<>
    void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spgemm_opsparse(Matrix_d &A, Matrix_d &B, Matrix_d &C)
    {
        // Make C "mutable".
        C.set_initialized(0);
        // Compute row offsets C.
        C.set_num_rows( A.get_num_rows() );
        C.set_num_cols( B.get_num_cols() );
        C.row_offsets.resize( A.get_num_rows() + 1 );
        C.m_seq_offsets.resize( A.get_num_rows() + 1 );
        thrust::sequence(C.m_seq_offsets.begin(), C.m_seq_offsets.end());
        cudaCheckError();

        CSR_opsparse _A, _B, _C;
        Meta meta;
        _A.M = A.get_num_rows();
        _A.N = A.get_num_cols();
        _A.nnz = A.get_num_nz();
        _A.d_rpt = (mint*) A.row_offsets.raw();
        _A.d_col = (mint*)A.col_indices.raw();
        _A.d_val = (mdouble*)A.values.raw();
        _B.M = B.get_num_rows();
        _B.N = B.get_num_cols();
        _B.nnz = B.get_num_nz();
        _B.d_rpt = (mint*)B.row_offsets.raw();
        _B.d_col = (mint*)B.col_indices.raw();
        _B.d_val = (mdouble*)B.values.raw();
        
        opsparse(_A, _B, _C, meta);
  
        int C_num_nnz1 = _C.nnz;
        C.col_indices.resize( C_num_nnz1 );
        C.diag.resize(C.get_num_rows());
        C.set_block_dimx(A.get_block_dimx());
        C.set_block_dimy(B.get_block_dimy());
        C.setColsReorderedByColor(false);
        C.set_num_nz( C_num_nnz1 );
        C.values.resize( C_num_nnz1 );

        cudaMemcpy(C.row_offsets.raw(), _C.d_rpt, (A.get_num_rows() + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(C.col_indices.raw(), _C.d_col, C_num_nnz1 * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(C.values.raw(), _C.d_val, C_num_nnz1 * sizeof(double), cudaMemcpyDeviceToDevice);

        _A.d_rpt = nullptr;
        _A.d_col = nullptr;
        _A.d_val = nullptr;
        _B.d_rpt = nullptr;
        _B.d_col = nullptr;
        _B.d_val = nullptr;

        // printf("A = %d %d %d\n", _A.M, _A.N, _A.nnz);
        // printf("B = %d %d %d\n", _B.M, _B.N, _B.nnz);
        // printf("C = %d %d %d\n", _C.M, _C.N, _C.nnz);
    
        C.set_initialized(1);
    }

*/

// /*
    template<>
    void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spgemm_opsparse(Matrix_d &A, Matrix_d &B, Matrix_d &C)
    {
        // Make C "mutable".
        C.set_initialized(0);
        // Compute row offsets C.
        C.set_num_rows( A.get_num_rows() );
        C.set_num_cols( B.get_num_cols() );
        C.row_offsets.resize( A.get_num_rows() + 1 );
        C.m_seq_offsets.resize( A.get_num_rows() + 1 );
        thrust::sequence(C.m_seq_offsets.begin(), C.m_seq_offsets.end());
        cudaCheckError();

        CSR_opsparse _A, _B, _C;
        Meta meta;
        _A.M = A.get_num_rows();
        _A.N = A.get_num_cols();
        _A.nnz = A.get_num_nz();
        _A.d_rpt = (mint*) A.row_offsets.raw();
        _A.d_col = (mint*)A.col_indices.raw();
        _A.d_val = (mdouble*)A.values.raw();
        _B.M = B.get_num_rows();
        _B.N = B.get_num_cols();
        _B.nnz = B.get_num_nz();
        _B.d_rpt = (mint*)B.row_offsets.raw();
        _B.d_col = (mint*)B.col_indices.raw();
        _B.d_val = (mdouble*)B.values.raw();
       
        // opsparse(_A, _B, _C, meta);
        _C.M = _A.M;
        _C.N = _B.N;
        _C.nnz = 0;
        // meta.allocate_rpt(C); // allocate C.rpt, other init procedure, default stream
        _C.d_rpt = C.row_offsets.raw();
        cudaMemset(_C.d_rpt + _C.M, 0, sizeof(mint));
        h_compute_flop(_A, _B, _C, meta); // compute flop, stream[0]
        meta.allocate(_C); // allocate other memory    
        CHECK_ERROR(cudaMemcpy(meta.max_row_nnz, _C.d_rpt + _C.M, sizeof(mint), cudaMemcpyDeviceToHost));
        CHECK_ERROR(cudaDeviceSynchronize());
        h_symbolic_binning(_C, meta);
        CHECK_ERROR(cudaDeviceSynchronize());
        // symbolic phase
        h_symbolic(_A, _B, _C, meta);
        CHECK_ERROR(cudaDeviceSynchronize());

        // numeric binning
        h_numeric_binning(_C, meta);
        CHECK_ERROR(cudaDeviceSynchronize());

        // malloc C
        _C.nnz = *meta.total_nnz;
        int C_num_nnz1 = _C.nnz;
        C.col_indices.resize( C_num_nnz1 );
        C.diag.resize(C.get_num_rows());
        C.set_block_dimx(A.get_block_dimx());
        C.set_block_dimy(B.get_block_dimy());
        C.setColsReorderedByColor(false);
        C.set_num_nz( C_num_nnz1 );
        C.values.resize( C_num_nnz1 );

        // printf("opsparse C.nnz = %d\n", C.nnz);

        // CHECK_ERROR(cudaMalloc(&C.d_val, C.nnz * sizeof(mdouble)));
        // CHECK_ERROR(cudaMalloc(&C.d_col, C.nnz * sizeof(mint)));
        _C.d_val = C.values.raw();
        _C.d_col = C.col_indices.raw();

        // prefix sum and malloc
        cub::DeviceScan::ExclusiveSum(meta.d_cub_storage, meta.cub_storage_size, _C.d_rpt, _C.d_rpt, _C.M + 1);
        CHECK_ERROR(cudaDeviceSynchronize());

        // numeric   
        h_numeric_full_occu(_A, _B, _C, meta);
        CHECK_ERROR(cudaDeviceSynchronize());

        // cleanup
        meta.release();
  
    
        // cudaMemcpy(C.row_offsets.raw(), _C.d_rpt, (A.get_num_rows() + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
        // cudaMemcpy(C.col_indices.raw(), _C.d_col, C_num_nnz1 * sizeof(int), cudaMemcpyDeviceToDevice);
        // cudaMemcpy(C.values.raw(), _C.d_val, C_num_nnz1 * sizeof(double), cudaMemcpyDeviceToDevice);

        _A.d_rpt = nullptr;
        _A.d_col = nullptr;
        _A.d_val = nullptr;
        _B.d_rpt = nullptr;
        _B.d_col = nullptr;
        _B.d_val = nullptr;
        _C.d_rpt = nullptr;
        _C.d_col = nullptr;
        _C.d_val = nullptr;

        // printf("A = %d %d %d\n", _A.M, _A.N, _A.nnz);
        // printf("B = %d %d %d\n", _B.M, _B.N, _B.nnz);
        // printf("C = %d %d %d\n", _C.M, _C.N, _C.nnz);
    
        C.set_initialized(1);
    }
// */

/*
    template<>
    void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spgemm_speck(Matrix_d &A, Matrix_d &B, Matrix_d &C)
    {
        // printf("spgemm_opsparse\n");
        // printf("A %d %d %d\n", A.get_num_rows(), A.get_num_cols(), A.get_num_nz());
        // printf("B %d %d %d\n", B.get_num_rows(), B.get_num_cols(), B.get_num_nz());
        // CSR_opsparse RR;
        // dCSR<double> dCsrHiRes, dCsrReference;

        // Make C "mutable".
        C.set_initialized(0);
        // Compute row offsets C.
        C.set_num_rows( A.get_num_rows() );
        C.set_num_cols( B.get_num_cols() );
        C.row_offsets.resize( A.get_num_rows() + 1 );
        C.m_seq_offsets.resize( A.get_num_rows() + 1 );
        thrust::sequence(C.m_seq_offsets.begin(), C.m_seq_offsets.end());
        cudaCheckError();

        dCSR<double> _A, _B, _C;
        _A.rows =  A.get_num_rows();
        _A.cols =  A.get_num_cols();
        _A.nnz = A.get_num_nz();
        _A.row_offsets = (unsigned int *) A.row_offsets.raw();
        _A.col_ids = (unsigned int *)A.col_indices.raw();
        _A.data = (double*)A.values.raw();
        
        _B.rows =  B.get_num_rows();
        _B.cols =  B.get_num_cols();
        _B.nnz =  B.get_num_nz();
        _B.row_offsets = (unsigned int *)B.row_offsets.raw();
        _B.col_ids = (unsigned int *)B.col_indices.raw();
        _B.data = (double*)B.values.raw();

        // printf("A = %d %d %d\n", _A.rows, _A.cols, _A.nnz);
        // printf("B = %d %d %d\n", _B.rows, _B.cols, _B.nnz);


        auto config = spECK::spECKConfig::initialize(0);
        Timings timings = Timings();
		timings.measureAll = false;
		timings.measureCompleteTime = false;
		spECK::MultiplyspECK<double, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>(_A, _B, _C, config, timings);
  
        int C_num_nnz1 = (int) _C.nnz;
        // printf("C_num_nnz1 = %d\n", C_num_nnz1);
        C.col_indices.resize( C_num_nnz1 );
        C.diag.resize(C.get_num_rows());
        C.set_block_dimx(A.get_block_dimx());
        C.set_block_dimy(B.get_block_dimy());
        C.setColsReorderedByColor(false);
        C.set_num_nz( C_num_nnz1 );
        C.values.resize( C_num_nnz1 );

        // printf("C = %d %d %d\n", _C.rows, _C.cols, _C.nnz);

        cudaMemcpy(C.row_offsets.raw(), _C.row_offsets, (A.get_num_rows() + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(C.col_indices.raw(), _C.col_ids, C_num_nnz1 * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(C.values.raw(), _C.data, C_num_nnz1 * sizeof(double), cudaMemcpyDeviceToDevice);

        _A.row_offsets = nullptr;
        _A.col_ids = nullptr;
        _A.data = nullptr;
        _B.row_offsets = nullptr;
        _B.col_ids = nullptr;
        _B.data = nullptr;

        // printf("A = %d %d %d\n", _A.M, _A.N, _A.nnz);
        // printf("B = %d %d %d\n", _B.M, _B.N, _B.nnz);
        // printf("C = %d %d %d\n", _C.M, _C.N, _C.nnz);
    
        C.set_initialized(1);
    }
*/

    template<>
    void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spgemm_speck(Matrix_d &A, Matrix_d &B, Matrix_d &C)
    {
        // printf("spgemm_speck\n");
        // printf("A %d %d %d\n", A.get_num_rows(), A.get_num_cols(), A.get_num_nz());
        // printf("B %d %d %d\n", B.get_num_rows(), B.get_num_cols(), B.get_num_nz());
        // CSR_opsparse RR;
        // dCSR<double> dCsrHiRes, dCsrReference;

        // Make C "mutable".
        C.set_initialized(0);
        // Compute row offsets C.
        C.set_num_rows( A.get_num_rows() );
        C.set_num_cols( B.get_num_cols() );
        C.row_offsets.resize( A.get_num_rows() + 1 );
        C.m_seq_offsets.resize( A.get_num_rows() + 1 );
        thrust::sequence(C.m_seq_offsets.begin(), C.m_seq_offsets.end());
        cudaCheckError();

        dCSR<double> _A, _B, matOut;
        _A.rows =  A.get_num_rows();
        _A.cols =  A.get_num_cols();
        _A.nnz = A.get_num_nz();
        _A.row_offsets = (unsigned int *) A.row_offsets.raw();
        _A.col_ids = (unsigned int *)A.col_indices.raw();
        _A.data = (double*)A.values.raw();
        
        _B.rows =  B.get_num_rows();
        _B.cols =  B.get_num_cols();
        _B.nnz =  B.get_num_nz();
        _B.row_offsets = (unsigned int *)B.row_offsets.raw();
        _B.col_ids = (unsigned int *)B.col_indices.raw();
        _B.data = (double*)B.values.raw();

        // printf("A = %d %d %d\n", _A.rows, _A.cols, _A.nnz);
        // printf("B = %d %d %d\n", _B.rows, _B.cols, _B.nnz);


        auto config = spECK::spECKConfig::initialize(0);
        Timings timings = Timings();
		timings.measureAll = false;
		timings.measureCompleteTime = false;
		
        //--------------------------------
        typedef double DataType;
        #define BLOCKS_PER_SM  4
        #define  THREADS_PER_BLOCK  1024
        #define  MAX_DYNAMIC_SHARED  spECK_DYNAMIC_MEM_PER_BLOCK
        #define  MAX_STATIC_SHARED  spECK_STATIC_MEM_PER_BLOCK

        // those matrices automatically deallocate memory when used as param for cuda -> therefore i have written a new struct without deallocs
        dCSRNoDealloc<DataType> matA(_A), matB(_B);

        if (matB.cols > 1 << 27)
        {
            printf("ERROR: matrix B has more than %d columns (%lu)\n", 1 << 27, matB.cols);
            return;
        }
        if (matA.rows > 1 << 27)
        {
            printf("ERROR: matrix A has more than %d rows (%lu)\n", 1 << 27, matB.rows);
            return;
        }
        if (matA.nnz * matB.nnz == 0) {
            matOut.nnz = 0;
            return;
        }

        if (MAX_DYNAMIC_SHARED != config.maxDynamicSharedMemoryPerBlock || MAX_STATIC_SHARED != config.maxStaticSharedMemoryPerBlock) {
            if (MAX_DYNAMIC_SHARED > config.maxDynamicSharedMemoryPerBlock) {
                printf("ERROR: spECK was compiled with %d maximum dynamic shared memory, but device limit is %d. Please recompile with correct amount set in Multiply.h line 10: spECK_DYNAMIC_MEM_PER_BLOCK\n",
                    MAX_DYNAMIC_SHARED, config.maxDynamicSharedMemoryPerBlock);
                return;
            } else {
                printf("WARNING: spECK was compiled with %d maximum dynamic shared memory, but device limit is %d. Please recompile with correct amount set in Multiply.h line 10: spECK_DYNAMIC_MEM_PER_BLOCK\n",
                    MAX_DYNAMIC_SHARED, config.maxDynamicSharedMemoryPerBlock);
            }
            if (MAX_STATIC_SHARED > MAX_DYNAMIC_SHARED)
            {
                printf("ERROR: spECK was compiled with smaller dynamic than static shared memory. (%d maximum static shared memory and %d maximum dynamic shared memory). Please check values in Multiply.h line 9 and 10",
                    MAX_STATIC_SHARED, MAX_DYNAMIC_SHARED);
                return;
            }
            if (MAX_STATIC_SHARED > config.maxStaticSharedMemoryPerBlock)
            {
                printf("ERROR: spECK was compiled with %d maximum static shared memory, but device limit is %d. Please recompile with correct amount set in Multiply.h line 9: spECK_STATIC_MEM_PER_BLOCK\n",
                    MAX_STATIC_SHARED, config.maxStaticSharedMemoryPerBlock);
                return;
            }
            else if (MAX_STATIC_SHARED < config.maxStaticSharedMemoryPerBlock) {
                printf("WARNING: spECK was compiled with %d maximum static shared memory, but device limit is %d. Please recompile with correct amount set in Multiply.h line 9: spECK_STATIC_MEM_PER_BLOCK\n",
                    MAX_STATIC_SHARED, config.maxStaticSharedMemoryPerBlock);
            }
        }

        // -------------------------------------------------------------------------------------------------------------------------------------------
        //  Constants and configs
        // -------------------------------------------------------------------------------------------------------------------------------------------

        spECKKernels spgemm(1024);

        const int kernelCountNumeric = 6;
        const int kernelCountCounting = 6;
        const int maxRowsPerBlock = 32; // this value may never exceed 32 because of some warp-optimizations
        const int warpsCounting = THREADS_PER_BLOCK / 32;
        const int warpsNumeric = THREADS_PER_BLOCK / 32;
        const int staticSharedMemPerBlockCounting = 48, staticSharedMemPerBlockNumeric = 24;
        const int sharedBytesPerWarpCounting = MAX_STATIC_SHARED / warpsCounting - staticSharedMemPerBlockCounting; // 48 byte is the maximum static shared memory per block
        const int entriesPerWarpCounting = sharedBytesPerWarpCounting / sizeof(IndexType);
        const int sharedBytesPerBlockCounting = sharedBytesPerWarpCounting * warpsCounting;
        // CC version > 7.0 support dynamic shared memory larger than static shared
        const int dynamicSharedBytesPerWarpCounting = MAX_DYNAMIC_SHARED / warpsCounting - staticSharedMemPerBlockCounting; // 48 byte is the maximum static shared memory per block
        const int dynamicEntriesPerWarpCounting = dynamicSharedBytesPerWarpCounting / sizeof(IndexType);
        const int dynamicSharedBytesPerBlockCounting = dynamicSharedBytesPerWarpCounting * warpsCounting;

        const int sharedBytesPerWarpNumeric = MAX_STATIC_SHARED / warpsNumeric - staticSharedMemPerBlockNumeric; // 24 byte is the maximum static shared memory per block
        const int entriesPerWarpNumeric = sharedBytesPerWarpNumeric / (sizeof(IndexType) + sizeof(DataType));
        const int sharedBytesPerBlockNumeric = sharedBytesPerWarpNumeric * warpsNumeric;
        // CC version > 7.0 support dynamic shared memory larger than static shared
        const int dynamicSharedBytesPerWarpNumeric = MAX_DYNAMIC_SHARED / warpsNumeric - staticSharedMemPerBlockNumeric; // 24 byte is the maximum static shared memory per block
        const int dynamicEntriesPerWarpNumeric = dynamicSharedBytesPerWarpNumeric / (sizeof(IndexType) + sizeof(DataType));
        const int dynamicSharedBytesPerBlockNumeric = dynamicSharedBytesPerWarpNumeric * warpsNumeric;
        assert(kernelCountCounting <= kernelCountNumeric);

        bool supportGlobalFallback = true;
        const uint32_t minimumDensityForDenseModeCounting = 999;
        const uint32_t denseModeRowThresholdInternalSorting = 999;
        const uint32_t denseModeRowThresholdExternalSorting = 18;
        const uint32_t sm = config.sm;
        const uint32_t cudaCores = config.sm * BLOCKS_PER_SM * 32;


        // -------------------------------------------------------------------------------------------------------------------------------------------
        //  INITIAL MALLOCS
        // -------------------------------------------------------------------------------------------------------------------------------------------

        int estimatedAvgComPerRow = max(1, int((matA.nnz / matA.rows) * (matB.nnz / matB.rows)));
        // determine how many nnz of matC should be calculated by one block. avoid hashmaps running full
        int maxNnzPerBlockNumeric = entriesPerWarpNumeric * warpsNumeric * 2 / 3;
        int maxNnzPerBlockNumericDynamicSharedMem = dynamicEntriesPerWarpNumeric * warpsNumeric * 2 / 3;

        // CUDA variables
        CUstream stream = config.streams[0];
        auto &streams = config.streams;
        
        if (timings.measureCompleteTime)
            startTimerVar(config.completeStart, stream);

        if (timings.measureAll)
            startTimerVar(config.individualStart, stream);

        // Allocate memory for offsets
        CU::unique_ptr newmat_offsets;
        // if (matOut.rows != matA.rows)
        // {
        //     newmat_offsets = CU::allocMemory((matA.rows + 1) * sizeof(IndexType));
        // }
        // else if (matOut.row_offsets != nullptr)
        {
            newmat_offsets.consume(reinterpret_cast<CUdeviceptr>(C.row_offsets.raw()));
            matOut.row_offsets = nullptr;
        }

        dCSRNoDealloc<DataType> matC;
        matC.row_offsets = newmat_offsets.get<IndexType>();
        matC.cols = matB.cols;
        matC.rows = matA.rows;

        IndexType *blockStartRowsScale = nullptr;
        IndexType *blockCounterScale = nullptr;
        IndexType h_blockCounterScaleNumeric[kernelCountNumeric] = {0};
        IndexType h_blockCounterScaleCounting[kernelCountCounting] = {0};

        size_t cubTempBytesScan = 0;
        size_t cubTmpBytesReduce = 0;
        size_t cubTmpBytesActual = 0;
        void *cubTmp = nullptr;

        {
            cub::DeviceScan::ExclusiveSum(cubTmp, cubTempBytesScan, matC.row_offsets, matC.row_offsets, matC.rows + 1);
            cub::DeviceReduce::Sum(cubTmp, cubTmpBytesReduce, matC.row_offsets, matC.row_offsets, matC.rows);
            cubTmpBytesReduce = std::max(cubTempBytesScan, cubTmpBytesReduce);
        }

        // ----------------------------------------------------------------------------------

        uint32_t maxComputationsPerRow = 0;
        uint32_t longestRowALength = 0;

        IndexType *d_blockStartRows = nullptr;
        uint32_t *d_blockCounter = nullptr;
        uint32_t *d_rowOperations = nullptr;
        uint32_t *d_rowMaxOperations = nullptr;
        uint32_t *d_maxElementsPerRow = nullptr;
        uint32_t *d_sumProducts = nullptr;
        uint32_t *d_rowColMinMax = nullptr;
        uint32_t *d_maxComputationsPerRow = nullptr;

        uint32_t *d_combined_pointers;
        size_t d_combined_pointers_size = sizeof(uint32_t) * (4 + 2 * matA.rows) + divup(cubTempBytesScan, sizeof(uint32_t)) * sizeof(uint32_t);
        if (matA.nnz > 10000)
            d_combined_pointers_size += sizeof(uint32_t) * matA.rows;

        HANDLE_ERROR(cudaMalloc(&d_combined_pointers, d_combined_pointers_size));
        HANDLE_ERROR(cudaMemsetAsync(d_combined_pointers, 0, d_combined_pointers_size));

        d_maxElementsPerRow = d_combined_pointers;
        /* keep this order */
        d_sumProducts = &d_maxElementsPerRow[1];
        d_maxComputationsPerRow = &d_sumProducts[1];
        /* until here */
        d_blockCounter = &d_maxComputationsPerRow[1];
        d_rowOperations = &d_blockCounter[1];
        d_rowMaxOperations = &d_rowOperations[matA.rows];
        cubTmp = (void *)&d_rowMaxOperations[matA.rows];
        cubTmpBytesActual = cubTempBytesScan;

        if (matA.nnz > 10000)
        {
            d_rowColMinMax = (uint32_t *)cubTmp;
            d_rowColMinMax = &d_rowColMinMax[divup(cubTempBytesScan, sizeof(uint32_t))];
        }

        if (timings.measureAll)
        {
            timings.init = recordTimerVar(config.individualStart, config.individualEnd, stream);
            startTimerVar(config.individualStart, stream);
        }

        // -------------------------------------------------------------------------------------------------------------------------------------------
        //  COUNT COMPUTATIONS
        // -------------------------------------------------------------------------------------------------------------------------------------------

        uint32_t sumProducts = 0;
        // calc amount of operations per row
        {
            const uint32_t threadsPerBlock = 128U;
            // limit to threadsPerBlock rows!
            // -> and always try to stay slightly below the threads per block size, because if you are slightly above, it is way more expensive than being far below
            uint32_t rowsPerBlock = std::min(threadsPerBlock, std::max(1U, (threadsPerBlock - 8) / std::max(1U, uint32_t(matA.nnz / matA.rows))));
            rowsPerBlock = std::max(1U, std::min(rowsPerBlock, uint32_t(matA.rows) / (4U * cudaCores / threadsPerBlock)));
            readOperations<IndexType, DataType, IndexType, threadsPerBlock><<<divup(uint32_t(matA.rows), rowsPerBlock), threadsPerBlock>>>(
                matA, matB, d_rowOperations, rowsPerBlock, d_maxComputationsPerRow, d_rowColMinMax, d_rowMaxOperations, d_sumProducts);

            // copying both values at once gives a huge performance boost
            uint32_t tmpArr[2];
            HANDLE_ERROR(cudaMemcpy(&tmpArr, d_sumProducts, sizeof(uint32_t) * 2, cudaMemcpyDeviceToHost));
            sumProducts = tmpArr[0];
            maxComputationsPerRow = tmpArr[1];
            // sumProducts = max(sumProducts, 1);
        }

        if (sumProducts == 0) {
            if (timings.measureCompleteTime)
                timings.complete = recordTimerVar(config.completeStart, config.completeEnd);
            // matOut.alloc(matA.rows, matB.cols, 0, false);
            matOut.rows = matC.rows;
            matOut.cols = matC.cols;
            matOut.nnz = sumProducts;
            
            // cudaMalloc(&data, sizeof(T)*n);
            // cudaMalloc(&col_ids, sizeof(unsigned int)*n);

            C.col_indices.resize( matOut.nnz );
            C.values.resize( matOut.nnz );
            matOut.data = C.values.raw();
            matOut.col_ids = (unsigned int *)C.col_indices.raw();
            return;
        }

        int maxNnzPerBlockCounting = entriesPerWarpCounting * warpsCounting * 4 / 5;
        int maxNnzPerBlockCountingDynamicSharedMem = dynamicEntriesPerWarpCounting * warpsCounting * 4 / 5;

        // you always know the maximum size of the output row
        uint32_t maxRowLength = max(1, min((uint32_t)matB.cols * 12 / 10, maxComputationsPerRow));

        if (timings.measureAll)
        {
            timings.countProducts = recordTimerVar(config.individualStart, config.individualEnd, stream);
            startTimerVar(config.individualStart, stream);
        }

        // -------------------------------------------------------------------------------------------------------------------------------------------
        //  LOADBALANCE COUNTING
        // -------------------------------------------------------------------------------------------------------------------------------------------

        uint32_t h_blockCounter = 0;

        uint32_t rowsPerBlock = 1;
        if (kernelCountCounting > 5 && maxRowLength < (maxNnzPerBlockCounting >> 4)) {
            uint32_t maxRowsPerBlockUtilization = max(1, min(uint32_t(maxRowsPerBlock), uint32_t(matA.rows / (sm * BLOCKS_PER_SM << (kernelCountCounting - 2)))));
            if (maxRowLength < maxNnzPerBlockCounting >> (kernelCountCounting - 1))
            {
                if (estimatedAvgComPerRow / maxRowLength == 1 || maxRowLength / estimatedAvgComPerRow == 1)
                    rowsPerBlock = min(maxRowsPerBlockUtilization, ((maxNnzPerBlockCounting >> (kernelCountCounting - 1)) / 3) / maxRowLength);
                else
                    rowsPerBlock = min(maxRowsPerBlockUtilization, (maxNnzPerBlockCounting >> kernelCountCounting) / maxRowLength);
            }
            rowsPerBlock = max(rowsPerBlock, 1);
            h_blockCounterScaleCounting[kernelCountCounting - 1] = divup(uint32_t(matA.rows), rowsPerBlock);
        }
        else if (kernelCountCounting > 4 && maxRowLength < (maxNnzPerBlockCounting >> 3))
            h_blockCounterScaleCounting[4] = matA.rows;
        else if (kernelCountCounting > 3 && maxRowLength < (maxNnzPerBlockCounting >> 2))
            h_blockCounterScaleCounting[3] = matA.rows;
        else if (kernelCountCounting > 2 && maxRowLength < (maxNnzPerBlockCounting >> 1))
            h_blockCounterScaleCounting[2] = matA.rows;
        else if (kernelCountCounting > 1 && maxRowLength < (maxNnzPerBlockCounting >> 0))
            h_blockCounterScaleCounting[1] = matA.rows;
        else
            h_blockCounterScaleCounting[0] = matA.rows;
            
        uint32_t rowsRequiringGlobal = h_blockCounterScaleCounting[0];

        uint32_t actualKernelCount = min(kernelCountCounting,
                                        uint32_t(
                                            std::log2(
                                                divup(
                                                    int(maxRowLength),
                                                    min(
                                                        maxNnzPerBlockCounting >> (kernelCountCounting - 1),
                                                        maxNnzPerBlockNumeric >> (kernelCountNumeric - 1)))) +
                                            1));

        bool useLoadBalancingCounting = false;


        // TODO check if && maxComputationsPerRow > maxNnzPerBlockCounting / 8 can be removed
        if (matA.nnz > 771843 || 
            maxComputationsPerRow < maxNnzPerBlockCountingDynamicSharedMem && maxComputationsPerRow > (maxNnzPerBlockCounting >> 2) && matA.rows > 7575 ||
            maxComputationsPerRow > maxNnzPerBlockCountingDynamicSharedMem && sumProducts > 1940177 ||
            maxComputationsPerRow / max(1, int((sumProducts / matA.rows))) > 110 && sumProducts > 1164708)
            useLoadBalancingCounting = true;

        if (useLoadBalancingCounting)
        {
            size_t combinedBlockStartSize = sizeof(IndexType) * (1 + kernelCountCounting + matA.rows * (1 + actualKernelCount));

            HANDLE_ERROR(cudaMalloc(&d_blockStartRows, combinedBlockStartSize));
            blockStartRowsScale = &d_blockStartRows[matA.rows + 1];
            blockCounterScale = &blockStartRowsScale[actualKernelCount * matA.rows];
            HANDLE_ERROR(cudaMemset(blockCounterScale, 0, sizeof(IndexType) * kernelCountCounting));

            // load balance over amount of operations per row in A
            spgemm.h_AssignHashSpGEMMBlocksToRowsOfSameSizeOperations<uint32_t, DataType, uint8_t, kernelCountCounting>(
                matA, matB, d_rowOperations, blockStartRowsScale, blockCounterScale, h_blockCounterScaleCounting, d_blockStartRows,
                maxNnzPerBlockCounting, maxNnzPerBlockCountingDynamicSharedMem, maxRowsPerBlock, actualKernelCount, rowsRequiringGlobal);
        }
        else
        {
            h_blockCounter = matA.rows;
            d_blockStartRows = nullptr;
        }

        if (timings.measureAll)
        {
            timings.loadBalanceCounting = recordTimerVar(config.individualStart, config.individualEnd, stream);
            startTimerVar(config.individualStart, stream);
        }

        // -------------------------------------------------------------------------------------------------------------------------------------------
        //  ALLOCATE GLOBAL MAPS
        // -------------------------------------------------------------------------------------------------------------------------------------------

        int elementsPerMap = (std::max(maxRowLength, uint32_t(maxNnzPerBlockCountingDynamicSharedMem)) * 5) / 4;
        supportGlobalFallback &= maxRowLength > entriesPerWarpCounting * warpsCounting;

        typedef HashMap<uint32_t, DataType> GlobalMap;
        typedef HashMapNoValue<uint32_t, 1> GlobalMapRowOffsets;
        typedef HashMapNoValue<uint32_t, maxRowsPerBlock> GlobalMapNoValue;
        void *hashMaps = nullptr;
        IndexType *maps_indices = nullptr;
        DataType *maps_values = nullptr;
        uint32_t hashMapCount = 0;
        size_t globalMapMaxSize;
        globalMapMaxSize = std::max(sizeof(GlobalMap), sizeof(GlobalMapNoValue));
        globalMapMaxSize = std::max(globalMapMaxSize, sizeof(GlobalMapRowOffsets));

        if (supportGlobalFallback)
        {
            hashMapCount = std::min(sm * BLOCKS_PER_SM, h_blockCounterScaleCounting[0]);
            hashMapCount = std::min(hashMapCount, rowsRequiringGlobal);
            supportGlobalFallback &= hashMapCount > 0;
        }

        rowsRequiringGlobal = matB.cols < entriesPerWarpCounting * warpsCounting ? 0 : rowsRequiringGlobal;
        bool isDenseCounting = useLoadBalancingCounting && rowsRequiringGlobal > 0 && maxComputationsPerRow > maxNnzPerBlockCountingDynamicSharedMem * 2;

        if (isDenseCounting)
        {
            supportGlobalFallback = false;
            // every bit is one column
            if (matB.cols > (warpsCounting * sharedBytesPerWarpCounting * 8) / 2)
            {
                if (longestRowALength == 0)
                {
                    uint32_t *d_longestRowALength = nullptr;
                    HANDLE_ERROR(cudaMalloc(&d_longestRowALength, sizeof(uint32_t)));
                    HANDLE_ERROR(cudaMemset(d_longestRowALength, 0, sizeof(uint32_t)));

                    const uint32_t blockdim = 256;
                    const uint32_t rowsPerThread = 2;
                    const uint32_t blocks = divup(IndexType(matA.rows), blockdim * rowsPerThread);
                    getLongestRowA<IndexType, blockdim, rowsPerThread><<<blocks, blockdim>>>(matA.row_offsets, d_longestRowALength, matA.rows, matA.nnz);
                    cudaMemcpy(&longestRowALength, d_longestRowALength, sizeof(uint32_t), cudaMemcpyDeviceToHost);
                }

                // only use global maps if the row cursors can't be held in shared memory
                if (elementsPerMap * 2 > warpsCounting * entriesPerWarpCounting)
                {
                    hashMapCount = sm * BLOCKS_PER_SM;
                    elementsPerMap = longestRowALength * 5 / 4;

                    if (maps_indices != nullptr)
                        HANDLE_ERROR(cudaFree(maps_indices));
                    if (hashMaps != nullptr)
                        HANDLE_ERROR(cudaFree(hashMaps));

                    HANDLE_ERROR(cudaMalloc(&maps_indices, sizeof(uint32_t) * hashMapCount * (elementsPerMap + maxRowsPerBlock + 1)));
                    HANDLE_ERROR(cudaMalloc(&hashMaps, globalMapMaxSize * hashMapCount));

                    spgemm.setLaunchDimensions(hashMapCount, streams[0], 32 * warpsNumeric);
                    spgemm.h_InitializeGlobalMapsNoVal<GlobalMapRowOffsets, uint32_t>((GlobalMapRowOffsets *)hashMaps, hashMapCount, maps_indices, elementsPerMap, maxRowsPerBlock);
                }
            }
        }

        if (supportGlobalFallback)
        {
            HANDLE_ERROR(cudaMalloc(&hashMaps, globalMapMaxSize * hashMapCount));
            HANDLE_ERROR(cudaMalloc(&maps_indices, sizeof(IndexType) * hashMapCount * (elementsPerMap + maxRowsPerBlock + 1)));

            spgemm.setLaunchDimensions(hashMapCount, streams[0], 32 * warpsCounting);
            spgemm.h_InitializeGlobalMapsNoVal<GlobalMapNoValue, IndexType>((GlobalMapNoValue *)hashMaps, hashMapCount, maps_indices, elementsPerMap, maxRowsPerBlock);
        }

        if (timings.measureAll)
        {
            timings.globalMapsCounting = recordTimerVar(config.individualStart, config.individualEnd, stream);
            startTimerVar(config.individualStart, stream);
        }

        // -------------------------------------------------------------------------------------------------------------------------------------------
        //  PRE-COUNTING LOAD-OPTIMIZATION
        // -------------------------------------------------------------------------------------------------------------------------------------------

        IndexType blockPrefixScaled[kernelCountCounting] = {0};
        {
            uint32_t activeSM = h_blockCounterScaleCounting[0];
            // never go up to top level
            int firstXEmpty = h_blockCounterScaleCounting[0] == 0;
            bool foundFirstNonEmpty = h_blockCounterScaleCounting[0] != 0;
            for (int i = 1; i < kernelCountCounting; ++i)
            {
                blockPrefixScaled[i] = h_blockCounterScaleCounting[i - 1] + blockPrefixScaled[i - 1];
                activeSM += 2 * h_blockCounterScaleCounting[i] >> (i - 1);
                if (!foundFirstNonEmpty)
                {
                    if (h_blockCounterScaleCounting[i] == 0)
                        firstXEmpty++;
                    else
                        foundFirstNonEmpty = true;
                }
            }

            // avoid div by zero
            activeSM = max(activeSM, 1);

            if (activeSM < sm * BLOCKS_PER_SM)
            {
                int shiftUp = min(firstXEmpty, int(std::log2(sm * BLOCKS_PER_SM / activeSM)));

                if (shiftUp > 0)
                {
                    for (int i = 0; i < kernelCountCounting; i++)
                    {
                        if (i + shiftUp < kernelCountCounting)
                        {
                            h_blockCounterScaleCounting[i] = h_blockCounterScaleCounting[i + shiftUp];
                            blockPrefixScaled[i] = blockPrefixScaled[i + shiftUp];
                        }
                        else
                        {
                            h_blockCounterScaleCounting[i] = 0;
                            blockPrefixScaled[i] = h_blockCounter;
                        }
                    }
                }
            }
        }

        // -------------------------------------------------------------------------------------------------------------------------------------------
        //  COUNT NNZ PER ROW OF C
        // -------------------------------------------------------------------------------------------------------------------------------------------

        {
            if (h_blockCounterScaleCounting[0] > 0)
            {
                if (isDenseCounting)
                {
                    // this only uses 1 block per sm and therefore hash 50% occupancy, but better caching
                    spgemm.setLaunchDimensions(h_blockCounterScaleCounting[0], streams[0], (32 * warpsCounting >> 0), dynamicSharedBytesPerBlockCounting);
                    spgemm.h_DenseSpGEMMCount<IndexType, DataType, GlobalMapRowOffsets, dynamicSharedBytesPerBlockCounting, true, (32 * warpsCounting >> 0)>(
                        matA, matB, (GlobalMapRowOffsets *)hashMaps, hashMapCount, matC.row_offsets, d_blockStartRows + blockPrefixScaled[0],
                        d_rowOperations, h_blockCounterScaleCounting[0], d_rowColMinMax,
                        d_rowMaxOperations, d_maxElementsPerRow, rowsPerBlock);
                }
                else
                {
                    spgemm.setLaunchDimensions(h_blockCounterScaleCounting[0], streams[0], 32 * warpsCounting >> 0, dynamicSharedBytesPerBlockCounting);
                    spgemm.h_SpGEMMCountLauncher<IndexType, DataType, maxRowsPerBlock, GlobalMapNoValue, GlobalMapRowOffsets, dynamicSharedBytesPerBlockCounting, true, (32 * warpsCounting >> 0)>(
                        matA, matB, (GlobalMapNoValue *)hashMaps, hashMapCount, nullptr, 0, matC.row_offsets, d_rowOperations,
                        d_blockStartRows + blockPrefixScaled[0], h_blockCounterScaleCounting[0], d_rowColMinMax,
                        d_rowMaxOperations, minimumDensityForDenseModeCounting, d_maxElementsPerRow, rowsPerBlock);
                }
            }

            if (kernelCountCounting > 1 && h_blockCounterScaleCounting[1] > 0)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleCounting[1], streams[1], 32 * warpsCounting >> 0, sharedBytesPerBlockCounting >> 0);
                spgemm.h_SpGEMMCountLauncher<IndexType, DataType, maxRowsPerBlock, GlobalMapNoValue, GlobalMapRowOffsets, (sharedBytesPerBlockCounting >> 0), false, (32 * warpsCounting >> 0)>(
                    matA, matB, (GlobalMapNoValue *)hashMaps, hashMapCount, nullptr, 0, matC.row_offsets, d_rowOperations,
                    d_blockStartRows + blockPrefixScaled[1], h_blockCounterScaleCounting[1], d_rowColMinMax,
                    d_rowMaxOperations, minimumDensityForDenseModeCounting, d_maxElementsPerRow, rowsPerBlock);
            }

            if (kernelCountCounting > 2 && h_blockCounterScaleCounting[2] > 0)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleCounting[2], streams[2], (32 * warpsCounting >> 1), sharedBytesPerBlockCounting >> 1);
                spgemm.h_SpGEMMCountLauncher<IndexType, DataType, maxRowsPerBlock, GlobalMapNoValue, GlobalMapRowOffsets, (sharedBytesPerBlockCounting >> 1), false, (32 * warpsCounting >> 1)>(
                    matA, matB, (GlobalMapNoValue *)hashMaps, hashMapCount, nullptr, 0, matC.row_offsets, d_rowOperations,
                    d_blockStartRows + blockPrefixScaled[2], h_blockCounterScaleCounting[2], d_rowColMinMax,
                    d_rowMaxOperations, minimumDensityForDenseModeCounting, d_maxElementsPerRow, rowsPerBlock);
            }

            if (kernelCountCounting > 3 && h_blockCounterScaleCounting[3] > 0)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleCounting[3], streams[3], (32 * warpsCounting >> 2), sharedBytesPerBlockCounting >> 2);
                spgemm.h_SpGEMMCountLauncher<IndexType, DataType, maxRowsPerBlock, GlobalMapNoValue, GlobalMapRowOffsets, (sharedBytesPerBlockCounting >> 2), false, (32 * warpsCounting >> 2)>(
                    matA, matB, (GlobalMapNoValue *)hashMaps, hashMapCount, nullptr, 0, matC.row_offsets, d_rowOperations,
                    d_blockStartRows + blockPrefixScaled[3], h_blockCounterScaleCounting[3], d_rowColMinMax,
                    d_rowMaxOperations, minimumDensityForDenseModeCounting, d_maxElementsPerRow, rowsPerBlock);
            }

            if (kernelCountCounting > 4 && h_blockCounterScaleCounting[4] > 0)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleCounting[4], streams[4], 32 * warpsCounting >> 3, sharedBytesPerBlockCounting >> 3);
                spgemm.h_SpGEMMCountLauncher<IndexType, DataType, maxRowsPerBlock, GlobalMapNoValue, GlobalMapRowOffsets, (sharedBytesPerBlockCounting >> 3), false, (32 * warpsCounting >> 3)>(
                    matA, matB, (GlobalMapNoValue *)hashMaps, hashMapCount, nullptr, 0, matC.row_offsets, d_rowOperations,
                    d_blockStartRows + blockPrefixScaled[4], h_blockCounterScaleCounting[4], d_rowColMinMax,
                    d_rowMaxOperations, minimumDensityForDenseModeCounting, d_maxElementsPerRow, rowsPerBlock);
            }

            if (kernelCountCounting > 5 && h_blockCounterScaleCounting[5] > 0)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleCounting[5], streams[5], 32 * warpsCounting >> 4, sharedBytesPerBlockCounting >> 4);
                spgemm.h_SpGEMMCountLauncher<IndexType, DataType, maxRowsPerBlock, GlobalMapNoValue, GlobalMapRowOffsets, (sharedBytesPerBlockCounting >> 4), false, (32 * warpsCounting >> 4)>(
                    matA, matB, (GlobalMapNoValue *)hashMaps, hashMapCount, nullptr, 0, matC.row_offsets, d_rowOperations,
                    d_blockStartRows + blockPrefixScaled[5], h_blockCounterScaleCounting[5], d_rowColMinMax,
                    d_rowMaxOperations, minimumDensityForDenseModeCounting, d_maxElementsPerRow, rowsPerBlock);
            }
        }

        // -------------------------------------------------------------------------------------------------------------------------------------------
        //  SCAN ROW OFFSETS AND GET NNZ OF C
        // -------------------------------------------------------------------------------------------------------------------------------------------

        // now we need to allocate that memory for prefix scan and for finding the longest row
        if (cubTmpBytesActual < cubTempBytesScan)
        {
            cubTmpBytesActual = cubTempBytesScan;
            if (cubTmp != nullptr)
                HANDLE_ERROR(cudaFree(cubTmp));
            HANDLE_ERROR(cudaMalloc(&cubTmp, cubTmpBytesActual));
        }

        // prefix sum to get the starting ids of each row of mat C
        cub::DeviceScan::ExclusiveSum(cubTmp, cubTmpBytesActual, matC.row_offsets, matC.row_offsets, matC.rows + 1);
        {
            IndexType nnz;
            cudaMemcpy(&nnz, matC.row_offsets + matC.rows, sizeof(IndexType), cudaMemcpyDeviceToHost);
            matC.nnz = nnz;
        }

        if (timings.measureAll)
        {
            HANDLE_ERROR(cudaDeviceSynchronize());
            timings.spGEMMCounting = recordTimerVar(config.individualStart, config.individualEnd, stream);
            startTimerVar(config.individualStart, stream);
        }

        // -------------------------------------------------------------------------------------------------------------------------------------------
        //  ALLOCATE OUTPUT MATRIX C
        // -------------------------------------------------------------------------------------------------------------------------------------------

        // only allocate mem for mat C if size is not correct
        // printf("matC.rows, matC.cols, matC.nnz = %d %d %d", matC.rows, matC.cols, matC.nnz);
        if (matOut.nnz != matC.nnz)
        {
            // matOut.alloc(matC.rows, matC.cols, matC.nnz, false);
            matOut.rows = matC.rows;
            matOut.cols = matC.cols;
            matOut.nnz = matC.nnz;
            
            // cudaMalloc(&data, sizeof(T)*n);
            // cudaMalloc(&col_ids, sizeof(unsigned int)*n);

            C.col_indices.resize( matOut.nnz );
            C.values.resize( matOut.nnz );
            matOut.data = C.values.raw();
            matOut.col_ids = (unsigned int *)C.col_indices.raw();
        }

        if (matOut.data == nullptr || matOut.col_ids == nullptr)
        {
            if (matOut.nnz > 0)
                printf("ERROR: out of memory\n");
            return;
        }

        matOut.row_offsets = std::move(newmat_offsets.getRelease<IndexType>());
        matC = dCSRNoDealloc<DataType>(matOut);

        if (timings.measureAll)
        {
            timings.allocC = recordTimerVar(config.individualStart, config.individualEnd, stream);
            startTimerVar(config.individualStart, stream);
        }

        // -------------------------------------------------------------------------------------------------------------------------------------------
        //  LOAD BALANCE NUMERIC
        // -------------------------------------------------------------------------------------------------------------------------------------------

        uint32_t maxElementsPerRow = maxRowLength;
        cudaMemcpy(&maxElementsPerRow, d_maxElementsPerRow, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        bool reprocessLoadBalanceNumeric = useLoadBalancingCounting;
        rowsPerBlock = 1;
        
        // get the longest row in order to minimize the global map size which needs to be allocated

        if (kernelCountNumeric > 5 && maxElementsPerRow < (maxNnzPerBlockNumeric >> 4)) {
            uint32_t maxRowsPerBlockUtilization = max(1, min(uint32_t(maxRowsPerBlock), uint32_t(matA.rows / (sm * BLOCKS_PER_SM << (kernelCountNumeric - 2)))));
            if (maxElementsPerRow<(entriesPerWarpNumeric * warpsNumeric)>> kernelCountNumeric)
            {
                if (maxElementsPerRow / max(1U, uint32_t(matC.nnz / matC.rows)) == 1)
                    rowsPerBlock = min(maxRowsPerBlockUtilization, (maxNnzPerBlockNumeric >> (kernelCountNumeric - 1)) / maxElementsPerRow);
                else
                    rowsPerBlock = min(maxRowsPerBlockUtilization, (entriesPerWarpNumeric * warpsNumeric >> (kernelCountNumeric - 1)) / maxElementsPerRow);
            }
            rowsPerBlock = max(rowsPerBlock, 1);
            h_blockCounterScaleNumeric[kernelCountNumeric - 1] = divup(uint32_t(matA.rows), rowsPerBlock);
        }
        else if (kernelCountNumeric > 4 && maxElementsPerRow < (maxNnzPerBlockNumeric >> 3))
            h_blockCounterScaleNumeric[4] = matC.rows;
        else if (kernelCountNumeric > 3 && maxElementsPerRow < (maxNnzPerBlockNumeric >> 2))
            h_blockCounterScaleNumeric[3] = matC.rows;
        else if (kernelCountNumeric > 2 && maxElementsPerRow < (maxNnzPerBlockNumeric >> 1))
            h_blockCounterScaleNumeric[2] = matC.rows;
        else if (kernelCountNumeric > 1 && maxElementsPerRow < (maxNnzPerBlockNumeric >> 0))
            h_blockCounterScaleNumeric[1] = matC.rows;
        else
            h_blockCounterScaleNumeric[0] = matC.rows;

        supportGlobalFallback = true;
        supportGlobalFallback &= maxElementsPerRow >= maxNnzPerBlockNumericDynamicSharedMem;
        rowsRequiringGlobal = h_blockCounterScaleNumeric[0];

        uint32_t avgElementsPerRow = max(1, int(matC.nnz / matC.rows));
        uint32_t maxAvgElementsPerRowRatio = maxElementsPerRow / avgElementsPerRow;
        reprocessLoadBalanceNumeric = false;
        if (maxElementsPerRow > (maxNnzPerBlockNumeric >> 2) && matA.rows >= 1236 && sumProducts > 636293 ||
            maxElementsPerRow > (maxNnzPerBlockNumeric >> (kernelCountNumeric - 1)) && (
                maxAvgElementsPerRowRatio > 4 && sumProducts > 4921876 ||
                maxAvgElementsPerRowRatio > 13 && sumProducts > 385847 ||
                maxAvgElementsPerRowRatio > 18 && sumProducts > 26263 && avgElementsPerRow > 22 ||
                maxAvgElementsPerRowRatio > 146))
            reprocessLoadBalanceNumeric = true;

        // can bring a performance benefit for some matrices, but has small overhead
        if (reprocessLoadBalanceNumeric && matC.nnz > 0)
        {
            if (d_blockCounter == nullptr)
            {
                HANDLE_ERROR(cudaMalloc(&d_blockCounter, sizeof(uint32_t)));
            }
            if (blockCounterScale == nullptr)
            {
                size_t combinedBlockStartSize = sizeof(IndexType) * (1 + kernelCountNumeric + matA.rows * (1 + actualKernelCount));

                HANDLE_ERROR(cudaMalloc(&d_blockStartRows, combinedBlockStartSize));
                blockStartRowsScale = &d_blockStartRows[matA.rows + 1];
                blockCounterScale = &blockStartRowsScale[actualKernelCount * matA.rows];
            }
            // reset buffers
            HANDLE_ERROR(cudaMemsetAsync(d_blockCounter, 0, sizeof(uint32_t)));
            HANDLE_ERROR(cudaMemsetAsync(blockCounterScale, 0, sizeof(IndexType) * kernelCountNumeric));

            spgemm.h_AssignHashSpGEMMBlocksToRowsOfSameSize<IndexType, DataType, uint8_t, kernelCountNumeric>(
                matC, blockStartRowsScale, d_blockStartRows, blockCounterScale, h_blockCounterScaleNumeric,
                maxNnzPerBlockNumeric, maxNnzPerBlockNumericDynamicSharedMem, maxRowsPerBlock, actualKernelCount, rowsRequiringGlobal);
        }
        else
        {
            HANDLE_ERROR(cudaFree(d_blockStartRows));
            d_blockStartRows = nullptr;
        }

        if (timings.measureAll)
        {
            timings.loadBalanceNumeric = recordTimerVar(config.individualStart, config.individualEnd, stream);
            startTimerVar(config.individualStart, stream);
        }

        // -------------------------------------------------------------------------------------------------------------------------------------------
        //  ALLOCATE GLOBAL MAPS
        // -------------------------------------------------------------------------------------------------------------------------------------------

        // always disabled since we always use dense mode for large rows
        supportGlobalFallback = false;
        if (supportGlobalFallback)
        {
            // update elements per map now that we know the lengths of each row --> could save some global memory and therefore allocation time
            elementsPerMap = max(maxElementsPerRow, maxNnzPerBlockNumericDynamicSharedMem) * 3 / 2;
            supportGlobalFallback &= h_blockCounterScaleNumeric[0] > 0;
            hashMapCount = min(sm * BLOCKS_PER_SM, h_blockCounterScaleNumeric[0]);
            hashMapCount = min(hashMapCount, rowsRequiringGlobal);
            supportGlobalFallback &= hashMapCount > 0;
        }

        rowsRequiringGlobal = matB.cols < entriesPerWarpNumeric * warpsNumeric ? 0 : rowsRequiringGlobal;
        bool isDenseOutput = h_blockCounterScaleNumeric[0] > 0;

        GlobalMapRowOffsets *rowOffsetMaps = nullptr;
        IndexType *rowOffsetMapIndices = nullptr;
        uint32_t rowOffsetMapCount = 0;
        uint32_t rowOffsetMapElementsPer = 0;

        if (isDenseOutput)
        {
            if (longestRowALength == 0)
            {
                uint32_t *d_longestRowALength = nullptr;
                HANDLE_ERROR(cudaMalloc(&d_longestRowALength, sizeof(uint32_t)));
                HANDLE_ERROR(cudaMemset(d_longestRowALength, 0, sizeof(uint32_t)));

                const uint32_t _threads = 256;
                const uint32_t rowsPerThread = 2;
                const uint32_t blocks = divup(IndexType(matA.rows), _threads * rowsPerThread);
                getLongestRowA<IndexType, _threads, rowsPerThread><<<blocks, _threads>>>(matA.row_offsets, d_longestRowALength, matA.rows, matA.nnz);

                cudaMemcpy(&longestRowALength, d_longestRowALength, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            }

            rowOffsetMapElementsPer = longestRowALength;
            rowOffsetMapCount = min(h_blockCounterScaleNumeric[0], sm * BLOCKS_PER_SM);

            // only allocate global maps if row cursors can't be held in share memory
            if (elementsPerMap * 2 * sizeof(IndexType) > warpsNumeric * entriesPerWarpNumeric * (sizeof(IndexType) + sizeof(DataType)))
            {
                if (h_blockCounterScaleNumeric[0] != 0)
                {
                    if (rowOffsetMaps != nullptr)
                        HANDLE_ERROR(cudaFree(rowOffsetMaps));
                    HANDLE_ERROR(cudaMalloc(&rowOffsetMaps, globalMapMaxSize * rowOffsetMapCount));

                    if (rowOffsetMapIndices != nullptr)
                    {
                        HANDLE_ERROR(cudaFree(rowOffsetMapIndices));
                        rowOffsetMapIndices = nullptr;
                    }

                    if (rowOffsetMapIndices == nullptr)
                        HANDLE_ERROR(cudaMalloc(&rowOffsetMapIndices, sizeof(IndexType) * rowOffsetMapCount * (rowOffsetMapElementsPer + maxRowsPerBlock + 1)));

                    spgemm.setLaunchDimensions(rowOffsetMapCount, stream, 32 * warpsNumeric);
                    spgemm.h_InitializeGlobalMapsNoVal<GlobalMapRowOffsets, uint32_t>((GlobalMapRowOffsets *)rowOffsetMaps, rowOffsetMapCount, rowOffsetMapIndices, rowOffsetMapElementsPer, maxRowsPerBlock);
                }
            }
        }

        if (timings.measureAll)
        {
            timings.globalMapsNumeric = recordTimerVar(config.individualStart, config.individualEnd, stream);
            startTimerVar(config.individualStart, stream);
        }

        // -------------------------------------------------------------------------------------------------------------------------------------------
        //  PRE-NUMERIC LOAD OPTIMIZATIONS
        // -------------------------------------------------------------------------------------------------------------------------------------------

        // alloc indices for rows which shall be sorted by cub
        bool sortAllInplace = false;
        {
            {

                uint32_t activeSM = h_blockCounterScaleNumeric[0];
                // never go up to top level
                int firstXEmpty = 0;
                bool foundFirstNonEmpty = h_blockCounterScaleNumeric[0] != 0;
                for (int i = 1; i < kernelCountNumeric; ++i)
                {
                    blockPrefixScaled[i] = h_blockCounterScaleNumeric[i - 1] + blockPrefixScaled[i - 1];
                    activeSM += 2 * h_blockCounterScaleNumeric[i] >> (i - 1);
                    if (!foundFirstNonEmpty)
                    {
                        if (h_blockCounterScaleNumeric[i] == 0)
                            firstXEmpty++;
                        else
                            foundFirstNonEmpty = true;
                    }
                }

                // avoid div by zero
                activeSM = max(activeSM, 1);

                if (activeSM < sm * BLOCKS_PER_SM)
                {
                    int shiftUp = min(firstXEmpty, int(std::log2(sm * BLOCKS_PER_SM / activeSM)));

                    if (shiftUp > 0)
                    {
                        if (firstXEmpty >= 2)
                            sortAllInplace = true;

                        for (int i = 0; i < kernelCountNumeric; i++)
                        {
                            if (i + shiftUp < kernelCountNumeric)
                            {
                                h_blockCounterScaleNumeric[i] = h_blockCounterScaleNumeric[i + shiftUp];
                                blockPrefixScaled[i] = blockPrefixScaled[i + shiftUp];
                            }
                            else
                            {
                                h_blockCounterScaleNumeric[i] = 0;
                                blockPrefixScaled[i] = h_blockCounter;
                            }
                        }
                    }
                }
            }

            // inplace starts to be faster if the size of the maps is getting smaller
            Config::SortModes sortMode = Config::SortModes::CubSegmentedSort;

            const uint32_t entrySize = sizeof(IndexType) + sizeof(DataType);

            Config::SpGEMMMethods spGemmMethodNumeric = Config::AutoSpGEMM;


            // -------------------------------------------------------------------------------------------------------------------------------------------
            //  NUMERIC SPGEMM
            // -------------------------------------------------------------------------------------------------------------------------------------------

            if (h_blockCounterScaleNumeric[0] > 0)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[0], streams[0], 32 * warpsNumeric, dynamicSharedBytesPerBlockNumeric);
                spgemm.h_DenseSpGEMMNumeric<IndexType, DataType, GlobalMapRowOffsets, dynamicSharedBytesPerBlockNumeric, true, (32 * warpsNumeric)>(
                    matA, matB, matC, (GlobalMapRowOffsets *)rowOffsetMaps, rowOffsetMapCount,
                    d_blockStartRows, d_rowOperations, h_blockCounterScaleNumeric[0], d_rowColMinMax,
                    d_rowMaxOperations, false, rowsPerBlock);
            }

            sortMode = sortAllInplace ? Config::InPlace : Config::Separate;

            bool setSortingBit = sortAllInplace ? false : maxElementsPerRow >= 500;

            if (kernelCountNumeric > 1 && h_blockCounterScaleNumeric[1] > 0)
            {
                if (spGemmMethodNumeric == Config::AutoSpGEMM)
                {
                    spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[1], streams[1], (32 * warpsNumeric >> 0), (sharedBytesPerBlockNumeric >> 0));
                    spgemm.h_SpGEMMNumericLauncher<IndexType, DataType, GlobalMap, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 0), false, (32 * warpsNumeric >> 0)>(
                        matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount, rowOffsetMaps, rowOffsetMapCount,
                        d_blockStartRows + blockPrefixScaled[1], d_rowOperations,
                        sortMode,
                        h_blockCounterScaleNumeric[1], d_rowColMinMax, 
                        d_rowMaxOperations, denseModeRowThresholdExternalSorting, setSortingBit, rowsPerBlock);
                }
                else if (spGemmMethodNumeric == Config::DenseSpGEMM)
                {
                    spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[1], streams[1], 32 * warpsNumeric >> 0, (sharedBytesPerBlockNumeric >> 0));
                    spgemm.h_DenseSpGEMMNumeric<IndexType, DataType, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 0), false, (32 * warpsNumeric >> 0)>(
                        matA, matB, matC, (GlobalMapRowOffsets *)hashMaps, hashMapCount,
                        d_blockStartRows + blockPrefixScaled[1], d_rowOperations,
                        h_blockCounterScaleNumeric[1], d_rowColMinMax,
                        d_rowMaxOperations, setSortingBit, rowsPerBlock);
                }
                else
                {
                    spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[1], streams[1], 32 * warpsNumeric, (sharedBytesPerBlockNumeric >> 0));
                    spgemm.h_HashSpGEMMNumeric<IndexType, DataType, GlobalMap, (sharedBytesPerBlockNumeric >> 0), false, (32 * warpsNumeric)>(
                        matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount,
                        d_blockStartRows + blockPrefixScaled[1], d_rowOperations,
                        sortMode,
                        h_blockCounterScaleNumeric[1], d_rowColMinMax,
                        d_rowMaxOperations, setSortingBit, rowsPerBlock);
                }
            }

            if (kernelCountNumeric > 2 && h_blockCounterScaleNumeric[2] > 0)
            {
                if (spGemmMethodNumeric == Config::AutoSpGEMM)
                {
                    spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[2], streams[2], (32 * warpsNumeric >> 1), (sharedBytesPerBlockNumeric >> 1));
                    spgemm.h_SpGEMMNumericLauncher<IndexType, DataType, GlobalMap, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 1), false, (32 * warpsNumeric >> 1)>(
                        matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount, rowOffsetMaps, rowOffsetMapCount,
                        d_blockStartRows + blockPrefixScaled[2], d_rowOperations,
                        sortMode,
                        h_blockCounterScaleNumeric[2], d_rowColMinMax,
                        d_rowMaxOperations, denseModeRowThresholdExternalSorting, setSortingBit, rowsPerBlock);
                }
                else if (spGemmMethodNumeric == Config::DenseSpGEMM)
                {
                    spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[2], streams[2], 32 * warpsNumeric >> 1, (sharedBytesPerBlockNumeric >> 1));
                    spgemm.h_DenseSpGEMMNumeric<IndexType, DataType, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 1), false, (32 * warpsNumeric >> 1)>(
                        matA, matB, matC, (GlobalMapRowOffsets *)hashMaps, hashMapCount,
                        d_blockStartRows + blockPrefixScaled[2], d_rowOperations,
                        h_blockCounterScaleNumeric[2], d_rowColMinMax,
                        d_rowMaxOperations, setSortingBit, rowsPerBlock);
                }
                else
                {
                    spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[2], streams[2], 32 * warpsNumeric >> 1, (sharedBytesPerBlockNumeric >> 1));
                    spgemm.h_HashSpGEMMNumeric<IndexType, DataType, GlobalMap, (sharedBytesPerBlockNumeric >> 1), false, (32 * warpsNumeric >> 1)>(
                        matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount,
                        d_blockStartRows + blockPrefixScaled[2], d_rowOperations,
                        sortMode,
                        h_blockCounterScaleNumeric[2], d_rowColMinMax,
                        d_rowMaxOperations, setSortingBit, rowsPerBlock);
                }
            }

            sortMode = Config::InPlace;

            if (kernelCountNumeric > 3 && h_blockCounterScaleNumeric[3] > 0)
            {
                if (spGemmMethodNumeric == Config::AutoSpGEMM)
                {
                    spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[3], streams[3], (32 * warpsNumeric >> 2), (sharedBytesPerBlockNumeric >> 2));
                    spgemm.h_SpGEMMNumericLauncher<IndexType, DataType, GlobalMap, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 2), false, (32 * warpsNumeric >> 2)>(
                        matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount, rowOffsetMaps, rowOffsetMapCount,
                        d_blockStartRows + blockPrefixScaled[3], d_rowOperations,
                        sortMode,
                        h_blockCounterScaleNumeric[3], d_rowColMinMax,
                        d_rowMaxOperations, denseModeRowThresholdInternalSorting, false, rowsPerBlock);
                }
                else if (spGemmMethodNumeric == Config::DenseSpGEMM)
                {
                    spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[3], streams[3], 32 * warpsNumeric >> 2, (sharedBytesPerBlockNumeric >> 2));
                    spgemm.h_DenseSpGEMMNumeric<IndexType, DataType, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 2), false, (32 * warpsNumeric >> 2)>(
                        matA, matB, matC, (GlobalMapRowOffsets *)hashMaps, hashMapCount,
                        d_blockStartRows + blockPrefixScaled[3], d_rowOperations,
                        h_blockCounterScaleNumeric[3], d_rowColMinMax,
                        d_rowMaxOperations, false, rowsPerBlock);
                }
                else
                {
                    spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[3], streams[3], 32 * warpsNumeric >> 2, (sharedBytesPerBlockNumeric >> 2));
                    spgemm.h_HashSpGEMMNumeric<IndexType, DataType, GlobalMap, (sharedBytesPerBlockNumeric >> 2), false, (32 * warpsNumeric >> 2)>(
                        matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount,
                        d_blockStartRows + blockPrefixScaled[3], d_rowOperations,
                        sortMode,
                        h_blockCounterScaleNumeric[3], d_rowColMinMax,
                        d_rowMaxOperations, false, rowsPerBlock);
                }
            }

            if (kernelCountNumeric > 4 && h_blockCounterScaleNumeric[4] > 0)
            {
                if (spGemmMethodNumeric == Config::AutoSpGEMM)
                {
                    spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[4], streams[4], (32 * warpsNumeric >> 3), (sharedBytesPerBlockNumeric >> 3));
                    spgemm.h_SpGEMMNumericLauncher<IndexType, DataType, GlobalMap, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 3), false, (32 * warpsNumeric >> 3)>(
                        matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount, rowOffsetMaps, rowOffsetMapCount,
                        d_blockStartRows + blockPrefixScaled[4], d_rowOperations,
                        sortMode,
                        h_blockCounterScaleNumeric[4], d_rowColMinMax,
                        d_rowMaxOperations, denseModeRowThresholdInternalSorting, false, rowsPerBlock);
                }
                else if (spGemmMethodNumeric == Config::DenseSpGEMM)
                {
                    spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[4], streams[4], 32 * warpsNumeric >> 3, (sharedBytesPerBlockNumeric >> 3));
                    spgemm.h_DenseSpGEMMNumeric<IndexType, DataType, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 3), false, (32 * warpsNumeric >> 3)>(
                        matA, matB, matC, (GlobalMapRowOffsets *)hashMaps, hashMapCount,
                        d_blockStartRows + blockPrefixScaled[4], d_rowOperations,
                        h_blockCounterScaleNumeric[4], d_rowColMinMax,
                        d_rowMaxOperations, false, rowsPerBlock);
                }
                else
                {
                    spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[4], streams[4], 32 * warpsNumeric >> 3, (sharedBytesPerBlockNumeric >> 3));
                    spgemm.h_HashSpGEMMNumeric<IndexType, DataType, GlobalMap, (sharedBytesPerBlockNumeric >> 3), false, (32 * warpsNumeric >> 3)>(
                        matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount,
                        d_blockStartRows + blockPrefixScaled[4], d_rowOperations,
                        sortMode,
                        h_blockCounterScaleNumeric[4], d_rowColMinMax,
                        d_rowMaxOperations, false, rowsPerBlock);
                }
            }

            if (kernelCountNumeric > 5 && h_blockCounterScaleNumeric[5] > 0)
            {
                if (spGemmMethodNumeric == Config::AutoSpGEMM || ((rowsPerBlock > 1 || reprocessLoadBalanceNumeric) && spGemmMethodNumeric != Config::HashSpGEMM))
                {
                    spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[5], streams[5], (32 * warpsNumeric >> 4), (sharedBytesPerBlockNumeric >> 4));
                    spgemm.h_SpGEMMNumericLauncher<IndexType, DataType, GlobalMap, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 4), false, (32 * warpsNumeric >> 4)>(
                        matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount, rowOffsetMaps, rowOffsetMapCount,
                        d_blockStartRows + blockPrefixScaled[5], d_rowOperations,
                        sortMode,
                        h_blockCounterScaleNumeric[5], d_rowColMinMax,
                        d_rowMaxOperations, denseModeRowThresholdInternalSorting, false, rowsPerBlock);
                }
                else if (spGemmMethodNumeric == Config::DenseSpGEMM)
                {

                    spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[5], streams[5], 32 * warpsNumeric >> 4, (sharedBytesPerBlockNumeric >> 4));
                    spgemm.h_DenseSpGEMMNumeric<IndexType, DataType, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 4), false, (32 * warpsNumeric >> 4)>(
                        matA, matB, matC, (GlobalMapRowOffsets *)hashMaps, hashMapCount,
                        d_blockStartRows + blockPrefixScaled[5], d_rowOperations,
                        h_blockCounterScaleNumeric[5], d_rowColMinMax,
                        d_rowMaxOperations, false, rowsPerBlock);
                }
                else
                {
                    spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[5], streams[5], 32 * warpsNumeric >> 4, (sharedBytesPerBlockNumeric >> 4));
                    spgemm.h_HashSpGEMMNumeric<IndexType, DataType, GlobalMap, (sharedBytesPerBlockNumeric >> 4), false, (32 * warpsNumeric >> 4)>(
                        matA, matB, matC, (GlobalMap *)hashMaps, hashMapCount,
                        d_blockStartRows + blockPrefixScaled[5], d_rowOperations,
                        sortMode,
                        h_blockCounterScaleNumeric[5], d_rowColMinMax,
                        d_rowMaxOperations, false, rowsPerBlock);
                }
            }
        }

        if (timings.measureAll)
        {
            HANDLE_ERROR(cudaDeviceSynchronize());
            timings.spGEMMNumeric = recordTimerVar(config.individualStart, config.individualEnd, stream);
            startTimerVar(config.individualStart, stream);
        }

        // -------------------------------------------------------------------------------------------------------------------------------------------
        //  SORT MEDIUM AND LONG ROWS
        // -------------------------------------------------------------------------------------------------------------------------------------------

        if (!sortAllInplace && (h_blockCounterScaleNumeric[1] + h_blockCounterScaleNumeric[2] > 0) && maxElementsPerRow >= 500)
        {
            if (h_blockCounterScaleNumeric[2] > 0)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[2], streams[2], 32 * warpsNumeric / 4);
                spgemm.h_HashSpGEMMSorting<uint32_t, DataType, 32 * warpsNumeric / 4, entriesPerWarpNumeric * 32 / 2>(
                    matC, d_blockStartRows + blockPrefixScaled[2], h_blockCounterScaleNumeric[2], true);
            }

            if (h_blockCounterScaleNumeric[1] > 0)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[1], streams[1], 32 * warpsNumeric / 2);
                spgemm.h_HashSpGEMMSorting<uint32_t, DataType, 32 * warpsNumeric / 2, entriesPerWarpNumeric * 32>(
                    matC, d_blockStartRows + blockPrefixScaled[1], h_blockCounterScaleNumeric[1], true);
            }
        }

        if (timings.measureAll)
        {
            HANDLE_ERROR(cudaDeviceSynchronize());
            timings.sorting = recordTimerVar(config.individualStart, config.individualEnd, stream);
            startTimerVar(config.individualStart, stream);
        }

        // -------------------------------------------------------------------------------------------------------------------------------------------
        //  FREE ALLOCATED MEMORY
        // -------------------------------------------------------------------------------------------------------------------------------------------

        if (d_blockStartRows != nullptr)
            HANDLE_ERROR(cudaFree(d_blockStartRows));
        if (hashMaps != nullptr)
            HANDLE_ERROR(cudaFree(hashMaps));
        if (maps_indices != nullptr)
            HANDLE_ERROR(cudaFree(maps_indices));
        if (maps_values != nullptr)
            HANDLE_ERROR(cudaFree(maps_values));

        if (d_combined_pointers != nullptr)
            HANDLE_ERROR(cudaFree(d_combined_pointers));

        if (rowOffsetMaps != nullptr)
            HANDLE_ERROR(cudaFree(rowOffsetMaps));
        if (rowOffsetMapIndices != nullptr)
            HANDLE_ERROR(cudaFree(rowOffsetMapIndices));

        if (timings.measureAll)
        {
            timings.cleanup = recordTimerVar(config.individualStart, config.individualEnd, stream);
        }

        // -------------------------------------------------------------------------------------------------------------------------------------------
        //  END
        // -------------------------------------------------------------------------------------------------------------------------------------------

        if (timings.measureCompleteTime) {
            HANDLE_ERROR(cudaDeviceSynchronize());
            timings.complete = recordTimerVar(config.completeStart, config.completeEnd, stream);
        }

        if (timings.measureAll)
        {
            /*printf("elements per global map=%d. mapCount=%d\n", elementsPerMap, hashMapCount);
            printf("matCNnz=%d, number of blocks = %d, %d, %d, %d, %d, %d\n", matC.nnz,
                h_blockCounterScaleNumeric[0],
                kernelCountNumeric > 1 ? h_blockCounterScaleNumeric[1] : -1,
                kernelCountNumeric > 2 ? h_blockCounterScaleNumeric[2] : -1,
                kernelCountNumeric > 3 ? h_blockCounterScaleNumeric[3] : -1,
                kernelCountNumeric > 4 ? h_blockCounterScaleNumeric[4] : -1,
                kernelCountNumeric > 5 ? h_blockCounterScaleNumeric[5] : -1);*/
            if (timings.measureAll)
            {
                printf("spECK     initial mallocs = %f ms\n", timings.init);
                printf("spECK  count computations = %f ms\n", timings.countProducts);
                printf("spECK       load-balancer = %f ms\n", timings.loadBalanceCounting);
                printf("spECK      GlobalMaps Cnt = %f ms\n", timings.globalMapsCounting);
                printf("spECK     counting kernel = %f ms\n", timings.spGEMMCounting);
                printf("spECK        malloc mat C = %f ms\n", timings.allocC);
                printf("spECK   num load-balancer = %f ms\n", timings.loadBalanceNumeric);
                printf("spECK     init GlobalMaps = %f ms\n", timings.globalMapsNumeric);
                printf("spECK      numeric kernel = %f ms\n", timings.spGEMMNumeric);
                printf("spECK      Sorting kernel = %f ms\n", timings.sorting);
                printf("spECK             cleanup = %f ms\n", timings.cleanup);
                printf("--------------------------------------------------------------\n");
            }
            if (timings.measureCompleteTime)
                printf("spECK            complete = %f ms\n\n", timings.complete);
        }

        matOut.rows = matC.rows;
        matOut.cols = matC.cols;
        matOut.nnz = matC.nnz;
        matOut.col_ids = matC.col_ids;
        matOut.row_offsets = matC.row_offsets;
        matOut.data = matC.data;

        //--------------------------------
        // spECK::MultiplyspECK<double, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>(_A, _B, _C, config, timings);
  
        int C_num_nnz1 = (int) matOut.nnz;
        // printf("C_num_nnz1 = %d\n", C_num_nnz1);
        // C.col_indices.resize( C_num_nnz1 );
        C.diag.resize(C.get_num_rows());
        C.set_block_dimx(A.get_block_dimx());
        C.set_block_dimy(B.get_block_dimy());
        C.setColsReorderedByColor(false);
        C.set_num_nz( C_num_nnz1 );
        // C.values.resize( C_num_nnz1 );

        // printf("C = %d %d %d\n", _C.rows, _C.cols, _C.nnz);

        // cudaMemcpy(C.row_offsets.raw(), _C.row_offsets, (A.get_num_rows() + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
        // cudaMemcpy(C.col_indices.raw(), _C.col_ids, C_num_nnz1 * sizeof(int), cudaMemcpyDeviceToDevice);
        // cudaMemcpy(C.values.raw(), _C.data, C_num_nnz1 * sizeof(double), cudaMemcpyDeviceToDevice);

        // TODO AMGX的Vector中定义了=符号，rhs如果是device_vector，其实也会进行memcpy
        // opSPARSE的乘法函数可以拆分，在符号阶段后对AMGX的矩阵数据结构扩容，然后在这个数据结构上操作
        // spECK则难拆分，在行指针分配内存的时候，定义了unique_handle类管理显存，
        
        // thrust::device_ptr<int> device_ptr((int*)_C.row_offsets);
        // thrust::device_vector<int> device_vec(device_ptr, device_ptr + A.get_num_rows() + 1);
        // C.row_offsets = device_vec;

        // thrust::device_ptr<int> device_ptr2((int*)_C.col_ids);
        // thrust::device_vector<int> device_vec2(device_ptr2, device_ptr2 + C_num_nnz1);
        // C.col_indices = static_cast<const cusp::array1d<int, cusp::device_memory>&>(device_vec2);

        // thrust::device_ptr<double> device_ptr3(_C.data);
        // thrust::device_vector<double> device_vec3(device_ptr3, device_ptr3 + C_num_nnz1);
        // C.values = device_vec3;

        _A.row_offsets = nullptr;
        _A.col_ids = nullptr;
        _A.data = nullptr;
        _B.row_offsets = nullptr;
        _B.col_ids = nullptr;
        _B.data = nullptr;
        matOut.row_offsets = nullptr;
        matOut.col_ids = nullptr;
        matOut.data = nullptr;

        // printf("A = %d %d %d\n", _A.M, _A.N, _A.nnz);
        // printf("B = %d %d %d\n", _B.M, _B.N, _B.nnz);
        // printf("C = %d %d %d\n", _C.M, _C.N, _C.nnz);
    
        C.set_initialized(1);
    }

    template<>
    void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spgemm_old(Matrix_d &A, Matrix_d &B, Matrix_d &C)
    {
        C.set_initialized(0);
        // Compute row offsets C.
        C.set_num_rows( A.get_num_rows() );
        C.set_num_cols( B.get_num_cols() );
        C.row_offsets.resize( A.get_num_rows() + 1 );
        C.m_seq_offsets.resize( A.get_num_rows() + 1 );
        thrust::sequence(C.m_seq_offsets.begin(), C.m_seq_offsets.end());
        cudaCheckError();

        Matrix_d _C;
        CSR_Multiply_Impl<TConfig_d> *impl = NULL;
        impl = static_cast<CSR_Multiply_Impl<TConfig_d> *>( new CSR_Multiply_Sm70<TConfig_d>() );
        assert( impl != NULL );

        impl->set_num_threads_per_row_count(2);
        impl->set_num_threads_per_row_compute(2);
            
        impl->multiply( A, B, _C, NULL, NULL, NULL, NULL );
        
        int C_num_nnz1 = (int) _C.get_num_nz();
        C.col_indices.resize( C_num_nnz1 );
        C.diag.resize(C.get_num_rows());
        C.set_block_dimx(A.get_block_dimx());
        C.set_block_dimy(B.get_block_dimy());
        C.setColsReorderedByColor(false);
        C.set_num_nz( C_num_nnz1 );
        C.values.resize( C_num_nnz1 );

        // printf("C = %d %d %d\n", _C.rows, _C.cols, _C.nnz);

        cudaMemcpy(C.row_offsets.raw(), _C.row_offsets.raw(), (A.get_num_rows() + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(C.col_indices.raw(), _C.col_indices.raw(), C_num_nnz1 * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(C.values.raw(), _C.values.raw(), C_num_nnz1 * sizeof(double), cudaMemcpyDeviceToDevice);

        C.set_initialized(1);

        delete impl;
    }

    template<>
    void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spgemm_upper(Matrix_d &A, Matrix_d &B, Matrix_d &C)
    {

        // 由于上限法基于了nsparse，这里暂时不好集成，只使用max_u1来决定是否使用opsparse验证一下加速比
        if(A.optalgo.max_mu > 47.47) {
            A.optalgo.spgemm_algo = 2;
            SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spgemm_opsparse(A, B, C);
            return;
        }

        C.set_initialized(0);
        // Compute row offsets C.
        C.set_num_rows( A.get_num_rows() );
        C.set_num_cols( B.get_num_cols() );
        C.row_offsets.resize( A.get_num_rows() + 1 );
        C.m_seq_offsets.resize( A.get_num_rows() + 1 );
        thrust::sequence(C.m_seq_offsets.begin(), C.m_seq_offsets.end());
        cudaCheckError();

        Matrix_d _C;
        CSR_Multiply_Impl<TConfig_d> *impl = NULL;
        impl = static_cast<CSR_Multiply_Impl<TConfig_d> *>( new CSR_Multiply_Sm70_upper<TConfig_d>() );
        assert( impl != NULL );

        int avg_nz_per_row = B.get_num_nz() / B.get_num_rows();

        // A P
        if(A.get_num_rows() == A.get_num_cols()){
            if ( avg_nz_per_row < 2 )
            {
                impl->set_num_threads_per_row_count(2);
                impl->set_num_threads_per_row_compute(2);
            }
            else
            {
                impl->set_num_threads_per_row_count(4);
                impl->set_num_threads_per_row_compute(4);
            }
        // R AP
        } else {
            impl->set_num_threads_per_row_count(avg_nz_per_row <= 16.0 ? 8 : 32);
            impl->set_num_threads_per_row_compute(32);
        }


        impl->multiply( A, B, _C, NULL, NULL, NULL, NULL );
        
        int C_num_nnz1 = (int) _C.get_num_nz();
        C.col_indices.resize( C_num_nnz1 );
        C.diag.resize(C.get_num_rows());
        C.set_block_dimx(A.get_block_dimx());
        C.set_block_dimy(B.get_block_dimy());
        C.setColsReorderedByColor(false);
        C.set_num_nz( C_num_nnz1 );
        C.values.resize( C_num_nnz1 );

        // printf("C = %d %d %d\n", _C.rows, _C.cols, _C.nnz);

        cudaMemcpy(C.row_offsets.raw(), _C.row_offsets.raw(), (A.get_num_rows() + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(C.col_indices.raw(), _C.col_indices.raw(), C_num_nnz1 * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(C.values.raw(), _C.values.raw(), C_num_nnz1 * sizeof(double), cudaMemcpyDeviceToDevice);

        C.set_initialized(1);

        delete impl;
    }

    template<> template<>
    void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &C)
    {
        gettimeofday(&tv0, NULL);

        if(A.optalgo.spmv_algo == -1) {
            A.optalgo.setup_3();
        }
        int spmv_algo = A.optalgo.spmv_algo;
        // printf("into spmv: %d\n", spmv_algo);
        switch(spmv_algo) {
            case 0:
                spmv_cusparse(A, B, C);
                break;
            case 1:
                spmv_scaler(A, B, C);
                break;
            case 2:
                spmv_adaptive(A, B, C);
                break;
            case -1:
                spmv_cusparse(A, B, C);
                break;
        }

        gettimeofday(&tv1, NULL);
        spmv_time += (tv1.tv_sec * 1000.0 + tv1.tv_usec / 1000.0)-(tv0.tv_sec * 1000.0 + tv0.tv_usec / 1000.0);
    }

    template<>
    void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spgemm( Matrix_d &A,  Matrix_d &B, Matrix_d &C)
    {
        gettimeofday(&tv0, NULL);
        
        // 复用第二层网格
        if(A.amg_level_index <= 1) {
            A.optalgo.setup_2(B.optalgo);
        }
        if (A.amg_level_index == 1) {
            if(A.get_num_rows() == A.get_num_cols()) SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::level2_AP = A.optalgo.spgemm_algo;
            else SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::level2_RAP = A.optalgo.spgemm_algo;
        }
        if(A.amg_level_index > 1) {
            if(A.get_num_rows() == A.get_num_cols()) A.optalgo.spgemm_algo = SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::level2_AP;
            else A.optalgo.spgemm_algo = SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::level2_RAP;
        }
        
        int spgemm_algo = A.optalgo.spgemm_algo;
        // printf("into spgemm: %d\n", spgemm_algo);
        switch(spgemm_algo) {
            case 0:
                SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spgemm_speck(A, B, C);
                break;
            case 1:
                SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spgemm_upper(A, B, C);
                break;
            case 2:
                SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spgemm_opsparse(A, B, C);
                break;
            case -1:
                SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spgemm_old(A, B, C);
                break;
        }

        gettimeofday(&tv1, NULL);
        spgemm_time += (tv1.tv_sec * 1000.0 + tv1.tv_usec / 1000.0)-(tv0.tv_sec * 1000.0 + tv0.tv_usec / 1000.0);
    }
}


namespace amgx {
    // 显式实例化类，因为AMGX确实会根据config调用一些不同精度的SpGEMM，所以必须显式实例化不同精度的类
    template class SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>;
    template class SpAMG<AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>;
    template class SpAMG<AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>;
    template class SpAMG<AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>;
    template class SpAMG<AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>;
    template class SpAMG<AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>;

    // 显式实例化类里面的SpMV方法，因为类模板的模板参数没用，所以我在替换SpMV算法的时候强行制定了AMGX_vecDouble, AMGX_matDouble, AMGX_indInt，只在函数模板的模板参数那里加上了，
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_scaler<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &C);
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_scaler<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>> &C);
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_scaler<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>> &C);
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_scaler<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>> &C);;
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_scaler<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>> &C);
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_scaler<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>> &C);

    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_cusparse<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &C);
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_cusparse<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>> &C);
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_cusparse<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>> &C);
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_cusparse<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>> &C);;
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_cusparse<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>> &C);
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_cusparse<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>> &C);

    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_adaptive<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &C);
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_adaptive<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>> &C);
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_adaptive<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>> &C);
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_adaptive<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>> &C);;
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_adaptive<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>> &C);
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_adaptive<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>> &C);

    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &C);
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>> &C);
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>> &C);
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>> &C);;
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>> &C);
    template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>> &C);

    // template void SpAMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>::spmv_scaler<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> &C);
    // template void SpAMG<AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>::spmv_scaler<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>> &C);
    // template void SpAMG<AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>::spmv_scaler<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>> &C);
    // template void SpAMG<AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>::spmv_scaler<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>> &C);;
    // template void SpAMG<AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>::spmv_scaler<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>> &C);
    // template void SpAMG<AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>::spmv_scaler<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>>(Matrix<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>> &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>> &B, Vector<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>> &C);
}


