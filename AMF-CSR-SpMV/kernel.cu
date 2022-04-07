#include "amf_csr.h"
#include <cuda_profiler_api.h>
#include <cusp/system/cuda/utils.h>
#include <cusp/system/cuda/arch.h>
#include <thrust/device_ptr.h>
#include <algorithm>

typedef typename csrDeviceMatrix::row_offsets_array_type::const_iterator     RowIterator;
typedef typename csrDeviceMatrix::column_indices_array_type::const_iterator  ColumnIterator;
typedef typename csrDeviceMatrix::values_array_type::const_iterator          ValueIterator1;
typedef typename valDeviceArray::const_iterator                            ValueIterator2;
typedef typename valDeviceArray::iterator                                  ValueIterator3;

#define FUNC_SHIFT_REG \
	   		(blockId ^ 0)? 0 : sum0 += tmp; \
	   		(blockId ^ 1)? 0 : sum1 += tmp; \
	   		(blockId ^ 2)? 0 : sum2 += tmp; \
	   		(blockId ^ 3)? 0 : sum3 += tmp; 

#define FUNC_VECTOR FUNC_SHIFT_REG
#define FUNC_ADAPTIVE FUNC_SHIFT_REG

template <  unsigned int THREADS_PER_VECTOR,
            unsigned int VECTORS_PER_BLOCK>
__global__ void RFCOC_spmv_adaptive_kernel(
    const IndexType num_blocks,
    const IndexType num_fold_rows,
    const RowIterator Ap,
    const ColumnIterator Aj,
    const ValueIterator1 Ax,
    const ValueIterator2 x,
    ValueIterator3 y )
{
    __shared__ volatile VALUE_TYPE sum[SHARED_MEM_SIZE];
    __shared__ volatile IndexType ptrs[VECTORS_PER_BLOCK][4];
    const IndexType s_block_id = (num_fold_rows + 1) * 2;
    for (IndexType block_id = blockIdx.x; block_id < num_blocks; block_id+=gridDim.x)
    {
        const IndexType fold_row_start = Ap[s_block_id + block_id];
        const IndexType fold_row_stop = Ap[s_block_id + block_id + 1];
        const IndexType rowIdx_start    = Ap[fold_row_start * 2];
        const IndexType rowPtr_start     = Ap[fold_row_start * 2 + 1];
        const IndexType rowIdx_stop     = Ap[fold_row_start * 2 + 2];
        const IndexType rowPtr_stop     = Ap[fold_row_start * 2 + 3];

        const IndexType rows_to_process = fold_row_stop - fold_row_start;
        const IndexType rows_to_fold  = rowIdx_stop - rowIdx_start;
        const IndexType nnz_to_fold = rowPtr_stop - rowPtr_start;

        if (rows_to_process > 1 || rows_to_fold > 1 || nnz_to_fold < SHARED_MEM_SIZE) // short rows
        {
            const IndexType thread_lane = threadIdx.x & (THREADS_PER_VECTOR-1);
            const IndexType vector_lane = threadIdx.x / THREADS_PER_VECTOR;
            const IndexType num_vectors = blockDim.x / THREADS_PER_VECTOR;

            for (IndexType row = fold_row_start + vector_lane; row < fold_row_stop; row += num_vectors)
            {
                if (THREADS_PER_VECTOR >= 4)
                {
                    if (thread_lane < 4)
                        ptrs[vector_lane][thread_lane] = Ap[row*2 + thread_lane];
                }
                else
                {
                    if (thread_lane == 0)
                    {
                        ptrs[vector_lane][0] = Ap[row * 2 + 0];
                        ptrs[vector_lane][1] = Ap[row * 2 + 1];
                        ptrs[vector_lane][2] = Ap[row * 2 + 2];
                        ptrs[vector_lane][3] = Ap[row * 2 + 3];
                    }
                }
                
                const IndexType rowIdx_start_0 = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
                const IndexType rowPtr_start_0 = ptrs[vector_lane][1];                   //same as: row_start = Ap[row];
                const IndexType rowIdx_stop_0  = ptrs[vector_lane][2];                   //same as: row_end   = Ap[row+1];
                const IndexType rowPtr_stop_0  = ptrs[vector_lane][3];                   //same as: row_end   = Ap[row+1];

                VALUE_TYPE sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0; 

                if (THREADS_PER_VECTOR == 32 && rowPtr_stop_0 - rowPtr_start_0 > 32)
                {
                    // ensure aligned memory access to Aj and Ax
                    IndexType jj = rowPtr_start_0 - (rowPtr_start_0 & (THREADS_PER_VECTOR - 1)) + thread_lane;

                    // accumulate local sums
                    if(jj >= rowPtr_start_0 && jj < rowPtr_stop_0) {
                        IndexType curColIdx = Aj[jj];
                        IndexType blockId = (curColIdx & FOLD_MASK);
                        IndexType colIdx = (curColIdx >> FOLD_BITS );
                        VALUE_TYPE tmp = Ax[jj] * x[colIdx];
                        FUNC_ADAPTIVE
                    }

                    // accumulate local sums
                    for(jj += THREADS_PER_VECTOR; jj < rowPtr_stop_0; jj += THREADS_PER_VECTOR) {
                        IndexType curColIdx = Aj[jj];
                        IndexType blockId = (curColIdx & FOLD_MASK);
                        IndexType colIdx = (curColIdx >> FOLD_BITS );
                        VALUE_TYPE tmp = Ax[jj] * x[colIdx];
                        FUNC_ADAPTIVE
                    }
                }
                else
                {
                    // accumulate local sums
                    for(IndexType jj = rowPtr_start_0 + thread_lane; jj < rowPtr_stop_0; jj += THREADS_PER_VECTOR) {
                        IndexType curColIdx = Aj[jj];
                        IndexType blockId = (curColIdx & FOLD_MASK);
                        IndexType colIdx = (curColIdx >> FOLD_BITS );
                        VALUE_TYPE tmp = Ax[jj] * x[colIdx];
                        FUNC_ADAPTIVE
                    }
                }

                const IndexType index_sum0 = vector_lane * THREADS_PER_VECTOR * FOLDGRAIN + thread_lane;
                const IndexType index_sum1 = index_sum0 + THREADS_PER_VECTOR;
                const IndexType index_sum2 = index_sum1 + THREADS_PER_VECTOR;
                const IndexType index_sum3 = index_sum2 + THREADS_PER_VECTOR;

                sum[index_sum0]  = sum0;
                sum[index_sum1]  = sum1;
                sum[index_sum2]  = sum2;
                sum[index_sum3]  = sum3;

                for (IndexType stride = THREADS_PER_VECTOR/2; stride > 0; stride/=2)
                {
                    if (thread_lane < stride)
                    {
                        sum[index_sum0] += sum[index_sum0 + stride];
                        sum[index_sum1] += sum[index_sum1 + stride];
                        sum[index_sum2] += sum[index_sum2 + stride];
                        sum[index_sum3] += sum[index_sum3 + stride];
                    }
                }
                
                // write result to global memory
                if (THREADS_PER_VECTOR >= 4)
                {
                    if (thread_lane < rowIdx_stop_0 - rowIdx_start_0)
                        y[rowIdx_start_0 + thread_lane] = VALUE_TYPE(sum[vector_lane * THREADS_PER_VECTOR * FOLDGRAIN + thread_lane * THREADS_PER_VECTOR]);
                }
                else 
                {
                    if(thread_lane == 0)
                    {
                        for (IndexType i = 0; i < rowIdx_stop_0 - rowIdx_start_0; i++)
                            y[rowIdx_start_0 + i] = VALUE_TYPE(sum[vector_lane * THREADS_PER_VECTOR * FOLDGRAIN + i * THREADS_PER_VECTOR]);
                    }
                }
            }
        }
        else // long row
        {
            sum[threadIdx.x] = 0.0;
            for (IndexType i = rowPtr_start + threadIdx.x; i < rowPtr_stop; i+=blockDim.x)
                sum[threadIdx.x] += Ax[i] * x[Aj[i]];


            for (IndexType stride = blockDim.x/2; stride > 0; stride /= 2)
            {
                __syncthreads();
                if (threadIdx.x < stride)
                    sum[threadIdx.x] += sum[threadIdx.x+stride];
            }

            if (threadIdx.x == 0)
                y[rowIdx_start] = VALUE_TYPE(sum[0]);
        }
        __syncthreads();
    }
}

template <  unsigned int THREADS_PER_VECTOR,
            unsigned int VECTORS_PER_BLOCK>
__global__ void RFCOC_spmv_vector_kernel(
    const IndexType num_fold_rows,
    const RowIterator Ap,
    const ColumnIterator Aj,
    const ValueIterator1 Ax,
    const ValueIterator2 x,
    ValueIterator3 y )
{
    __shared__ volatile VALUE_TYPE sum[MAX_FOLD_ROWS][VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR/2];  // padded to avoid reduction conditionals
    __shared__ volatile IndexType ptrs[VECTORS_PER_BLOCK][4];

    const IndexType thread_id   = blockDim.x * blockIdx.x + threadIdx.x;    // global thread index
    const IndexType thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const IndexType vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const IndexType vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const IndexType num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors
 
    for(IndexType row = vector_id; row < num_fold_rows; row += num_vectors)
    {
        // use four threads to fetch row indexes and row pointers
        if (THREADS_PER_VECTOR >= 4)
        {
            if(thread_lane < 4)
                ptrs[vector_lane][thread_lane] = Ap[row * 2 + thread_lane];
        }
        else
        {
            if (thread_lane == 0) {
                ptrs[vector_lane][0] = Ap[row * 2 + 0];
                ptrs[vector_lane][1] = Ap[row * 2 + 1];
                ptrs[vector_lane][2] = Ap[row * 2 + 2];
                ptrs[vector_lane][3] = Ap[row * 2 + 3];
            }
        }

        const IndexType rowIdx_start = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
        const IndexType row_start    = ptrs[vector_lane][1];                   //same as: row_start = Ap[row];
        const IndexType rowIdx_end   = ptrs[vector_lane][2];                   //same as: row_end   = Ap[row+1];
        const IndexType row_end      = ptrs[vector_lane][3];                   //same as: row_end   = Ap[row+1];

        VALUE_TYPE sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0; 
		
        if (THREADS_PER_VECTOR == 32 && row_end - row_start > 32)
        {
           // ensure aligned memory access to Aj and Ax
           IndexType jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;

           // accumulate local sums
           if(jj >= row_start && jj < row_end) 
           {
                IndexType curColIdx = Aj[jj];
                IndexType blockId = (curColIdx & FOLD_MASK);
                IndexType colIdx = (curColIdx >> FOLD_BITS );
                VALUE_TYPE tmp = Ax[jj] * x[colIdx];
                FUNC_VECTOR
            }
            // accumulate local sums
            for(jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR) 
            {
                IndexType curColIdx = Aj[jj];
                IndexType blockId = (curColIdx & FOLD_MASK);
                IndexType colIdx = (curColIdx >> FOLD_BITS );
                VALUE_TYPE tmp = Ax[jj] * x[colIdx];
                FUNC_VECTOR
            }
        }
        else
        {
            // accumulate local sums
            for(IndexType jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR) 
            {
                IndexType curColIdx = Aj[jj];
                IndexType blockId = (curColIdx & FOLD_MASK);
                IndexType colIdx = (curColIdx >> FOLD_BITS );
                VALUE_TYPE tmp = Ax[jj] * x[colIdx];
                FUNC_VECTOR
            }
        }

        VALUE_TYPE temp0, temp;
	    sum[0][threadIdx.x] = sum0;
	    sum[1][threadIdx.x] = sum1;
	    sum[2][threadIdx.x] = sum2;
	    sum[3][threadIdx.x] = sum3;

        // reduce local sums to row sum
        for (IndexType jj = 0; jj < rowIdx_end - rowIdx_start; jj++) 
        {
            temp0 = sum[jj][threadIdx.x];
            if (THREADS_PER_VECTOR > 16) {
                temp = sum[jj][threadIdx.x + 16];
                sum[jj][threadIdx.x] = temp0 = temp0 + temp;
            }
            if (THREADS_PER_VECTOR >  8) {
                temp = sum[jj][threadIdx.x + 8];
                sum[jj][threadIdx.x] = temp0 = temp0 + temp;
            }
            if (THREADS_PER_VECTOR >  4) {
                temp = sum[jj][threadIdx.x + 4];
                sum[jj][threadIdx.x] = temp0 = temp0 + temp;
            }
            if (THREADS_PER_VECTOR >  2) {
                temp = sum[jj][threadIdx.x + 2];
                sum[jj][threadIdx.x] = temp0 = temp0 + temp;
            }
            if (THREADS_PER_VECTOR >  1) {
                temp = sum[jj][threadIdx.x + 1];
                sum[jj][threadIdx.x] = temp0 = temp0 + temp;
            }
        }
        if (THREADS_PER_VECTOR >= 4)
        {
            if (thread_lane < rowIdx_end - rowIdx_start)
                y[rowIdx_start + thread_lane] = VALUE_TYPE(sum[thread_lane][threadIdx.x - thread_lane]);
        }
        else
        {
            if (thread_lane == 0)  
            {
                for (IndexType row = 0; row < rowIdx_end - rowIdx_start; row++)
                    y[row + rowIdx_start] = VALUE_TYPE(sum[row][threadIdx.x]);
            }
        }
    }
}

template <unsigned int THREADS_PER_VECTOR>
void RFCOC_spmv_prepare1(
    struct RECORD* record1,
    csrDeviceMatrix &A,
    valDeviceArray &x,
    valDeviceArray &y)
{
    const IndexType THREADS_PER_BLOCK = 128;    // THREADS_PER_BLOCK*FOLDGRAIN <= SHARED_MEM_SIZE
    const IndexType VECTORS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
    
    if (record1->num_blocks == 1)   // regular matrix
    {
		const IndexType MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(RFCOC_spmv_vector_kernel<THREADS_PER_VECTOR, VECTORS_PER_BLOCK>, THREADS_PER_BLOCK, 0);
		const IndexType REQUIRED_BLOCKS = (record1->num_fold_rows + VECTORS_PER_BLOCK -1)/VECTORS_PER_BLOCK;
		const IndexType NUM_BLOCKS = std::min<IndexType>(MAX_BLOCKS, REQUIRED_BLOCKS);
        RFCOC_spmv_vector_kernel<THREADS_PER_VECTOR, VECTORS_PER_BLOCK><<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, 0>>>(
            record1->num_fold_rows, A.row_offsets.begin(), A.column_indices.begin(), A.values.begin(), x.begin(), y.begin());
    }
    else    // irregular matrix
    {
		const IndexType MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(RFCOC_spmv_adaptive_kernel<THREADS_PER_VECTOR, VECTORS_PER_BLOCK>, THREADS_PER_BLOCK, 0);
		const IndexType NUM_BLOCKS = std::min<IndexType>(MAX_BLOCKS, record1->num_blocks);
        RFCOC_spmv_adaptive_kernel<THREADS_PER_VECTOR, VECTORS_PER_BLOCK><<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, 0>>>(
            record1->num_blocks, record1->num_fold_rows, A.row_offsets.begin(), A.column_indices.begin(), A.values.begin(), x.begin(), y.begin());
    }
}

void RFCOC_spmv_prepare0(
    struct RECORD* record1,
    csrDeviceMatrix &A,
    valDeviceArray &x,
    valDeviceArray &y)
{
    IndexType TPV = record1->TPV;

    if (TPV <= 2)
    {
        RFCOC_spmv_prepare1<2>(record1, A, x, y);
        return;
    }
    else if (TPV <= 4)
    {
        RFCOC_spmv_prepare1<4>(record1, A, x, y);
        return;
    }
    else if (TPV <= 8)
    {
        RFCOC_spmv_prepare1<8>(record1, A, x, y);
        return;
    }
    else if (TPV <= 16)
    {
        RFCOC_spmv_prepare1<16>(record1, A, x, y);
        return;
    }

    RFCOC_spmv_prepare1<32>(record1, A, x, y);
}
