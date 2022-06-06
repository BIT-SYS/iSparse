/*
 *  Copyright 2008-2014 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#pragma once

#include <cusp/system/cuda/arch.h>
#include <cusp/system/cuda/utils.h>

#include <thrust/device_ptr.h>

#include <algorithm>

#include <cuda_profiler_api.h>

namespace cusp
{
namespace system
{
namespace cuda
{
namespace detail
{

//////////////////////////////////////////////////////////////////////////////
// CSR SpMV kernels based on a vector model (one warp per row)
//////////////////////////////////////////////////////////////////////////////
//
// spmv_csr_vector_device
//   Each row of the CSR matrix is assigned to a warp.  The warp computes
//   y[i] = A[i,:] * x, i.e. the dot product of the i-th row of A with
//   the x vector, in parallel.  This division of work implies that
//   the CSR index and data arrays (Aj and Ax) are accessed in a contiguous
//   manner (but generally not aligned).  On GT200 these accesses are
//   coalesced, unlike kernels based on the one-row-per-thread division of
//   work.  Since an entire 32-thread warp is assigned to each row, many
//   threads will remain idle when their row contains a small number
//   of elements.  This code relies on implicit synchronization among
//   threads in a warp.
//
// spmv_csr_vector_tex_device
//   Same as spmv_csr_vector_tex_device, except that the texture cache is
//   used for accessing the x vector.
//
//  Note: THREADS_PER_VECTOR must be one of [2,4,8,16,32]

#define MAX_FOLD_ROWS 4
// __device__ int fold_bits;
// __device__ int foldgrain_max_value;

template <typename RowIterator, typename ColumnIterator, typename ValueIterator1,
         typename ValueIterator2, typename ValueIterator3,
         typename UnaryFunction, typename BinaryFunction1, typename BinaryFunction2,
         unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__global__ void
spmv_csr_vector_kernel(const unsigned int num_rows,
                       const RowIterator    Ap,
                       const ColumnIterator Aj,
                       const ValueIterator1 Ax,
                       const ValueIterator2  x,
                       ValueIterator3        y,
                       UnaryFunction initialize,
                       BinaryFunction1 combine,
                       BinaryFunction2 reduce)
{
    typedef typename thrust::iterator_value<RowIterator>::type    IndexType;
    typedef typename thrust::iterator_value<ValueIterator1>::type ValueType;

     __shared__ volatile ValueType sdata[MAX_FOLD_ROWS][VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
    __shared__ volatile IndexType ptrs[VECTORS_PER_BLOCK][4];


    const IndexType THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    const IndexType thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const IndexType thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const IndexType vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const IndexType vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const IndexType num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

// 	if (thread_id == 0) {
// 		fold_bits = Ap[2 * (num_rows + 1)];
// 		foldgrain_max_value = Ap[2 * (num_rows +1) + 1];
// 	}
	IndexType fold_bits = Ap[2 * (num_rows + 1)];
	IndexType foldgrain_max_value = Ap[2 * (num_rows +1) + 1];
	// IndexType fold_bits = 4;
	// IndexType foldgrain_max_value = 15;

    for(IndexType row = vector_id; row < num_rows; row += num_vectors)
    {
        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version
        if(thread_lane < 4)
            ptrs[vector_lane][thread_lane] = Ap[row * 2 + thread_lane ];
            // ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];

        const IndexType rowIdx_start = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
        const IndexType row_start = ptrs[vector_lane][1];                   //same as: row_start = Ap[row];
        const IndexType rowIdx_end   = ptrs[vector_lane][2];                   //same as: row_end   = Ap[row+1];
        const IndexType row_end   = ptrs[vector_lane][3];                   //same as: row_end   = Ap[row+1];

// 		for (IndexType i = 0; i < MAX_FOLD_ROWS; i++) {
// 			sdata[i][threadIdx.x] = (ValueType)0.0;
// 		}

		ValueType sum[MAX_FOLD_ROWS] = {0};

        if (THREADS_PER_VECTOR == 32 && row_end - row_start > 32)
        {
            // ensure aligned memory access to Aj and Ax

            IndexType jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;

            // accumulate local sums
            if(jj >= row_start && jj < row_end) {
				IndexType curColIdx = Aj[jj];
				IndexType blockId = (curColIdx & foldgrain_max_value);
				IndexType colIdx = (curColIdx >> fold_bits );
				// sum[blockId] = reduce(sum[blockId], combine(Ax[jj], x[colIdx]));
				switch(blockId) {
					case 0: sum[0] = reduce(sum[0], combine(Ax[jj], x[colIdx])); break;
					case 1: sum[1] = reduce(sum[1], combine(Ax[jj], x[colIdx])); break;
					case 2: sum[2] = reduce(sum[2], combine(Ax[jj], x[colIdx])); break;
					case 3: sum[3] = reduce(sum[3], combine(Ax[jj], x[colIdx])); break;
					// case 4: sum[4] = reduce(sum[4], combine(Ax[jj], x[colIdx])); break;
					// case 5: sum[5] = reduce(sum[5], combine(Ax[jj], x[colIdx])); break;
					// case 6: sum[6] = reduce(sum[6], combine(Ax[jj], x[colIdx])); break;
					// case 7: sum[7] = reduce(sum[7], combine(Ax[jj], x[colIdx])); break;
					default: break;
				}
			}

            // accumulate local sums
            for(jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR) {
				IndexType curColIdx = Aj[jj];
				IndexType blockId = (curColIdx & foldgrain_max_value);
				IndexType colIdx = (curColIdx >> fold_bits );
				// sum[blockId] = reduce(sum[blockId], combine(Ax[jj], x[colIdx]));
				switch(blockId) {
					case 0: sum[0] = reduce(sum[0], combine(Ax[jj], x[colIdx])); break;
					case 1: sum[1] = reduce(sum[1], combine(Ax[jj], x[colIdx])); break;
					case 2: sum[2] = reduce(sum[2], combine(Ax[jj], x[colIdx])); break;
					case 3: sum[3] = reduce(sum[3], combine(Ax[jj], x[colIdx])); break;
					// case 4: sum[4] = reduce(sum[4], combine(Ax[jj], x[colIdx])); break;
					// case 5: sum[5] = reduce(sum[5], combine(Ax[jj], x[colIdx])); break;
					// case 6: sum[6] = reduce(sum[6], combine(Ax[jj], x[colIdx])); break;
					// case 7: sum[7] = reduce(sum[7], combine(Ax[jj], x[colIdx])); break;
					default: break;
				}
			}
        }
        else
        {
            // accumulate local sums
            for(IndexType jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR) {
				IndexType curColIdx = Aj[jj];
				IndexType blockId = (curColIdx & foldgrain_max_value);
				IndexType colIdx = (curColIdx >> fold_bits );
				// sum[blockId] = reduce(sum[blockId], combine(Ax[jj], x[colIdx]));
				switch(blockId) {
					case 0: sum[0] = reduce(sum[0], combine(Ax[jj], x[colIdx])); break;
					case 1: sum[1] = reduce(sum[1], combine(Ax[jj], x[colIdx])); break;
					case 2: sum[2] = reduce(sum[2], combine(Ax[jj], x[colIdx])); break;
					case 3: sum[3] = reduce(sum[3], combine(Ax[jj], x[colIdx])); break;
					// case 4: sum[4] = reduce(sum[4], combine(Ax[jj], x[colIdx])); break;
					// case 5: sum[5] = reduce(sum[5], combine(Ax[jj], x[colIdx])); break;
					// case 6: sum[6] = reduce(sum[6], combine(Ax[jj], x[colIdx])); break;
					// case 7: sum[7] = reduce(sum[7], combine(Ax[jj], x[colIdx])); break;
					default: break;
				}
			}
        }

        // TODO: remove temp var WAR for MSVC
		ValueType temp, temp0, temp1, temp2, temp3;
	
// #pragma unroll
		// for (IndexType jj = 0; jj < MAX_FOLD_ROWS; jj++) {
			// switch(jj) {
			// 	case 0: sdata[jj][threadIdx.x] = temp0 = sum[0]; break;
			// 	case 1: sdata[jj][threadIdx.x] = temp0 = sum[1]; break;
			// 	case 2: sdata[jj][threadIdx.x] = temp0 = sum[2]; break;
			// 	case 3: sdata[jj][threadIdx.x] = temp0 = sum[3]; break;
			// }
			// sdata[jj][threadIdx.x] = temp0 = sum[0];
			sdata[0][threadIdx.x] = temp0 = sum[0];
			sdata[1][threadIdx.x] = temp1 = sum[1];
			sdata[2][threadIdx.x] = temp2 = sum[2];
			sdata[3][threadIdx.x] = temp3 = sum[3];

    		if (THREADS_PER_VECTOR > 16) {
				temp = sdata[0][threadIdx.x + 16];
				sdata[0][threadIdx.x] = temp0 = reduce(temp0, temp);
				temp = sdata[1][threadIdx.x + 16];
				sdata[1][threadIdx.x] = temp1 = reduce(temp1, temp);
				temp = sdata[2][threadIdx.x + 16];
				sdata[2][threadIdx.x] = temp2 = reduce(temp2, temp);
				temp = sdata[3][threadIdx.x + 16];
				sdata[3][threadIdx.x] = temp3 = reduce(temp3, temp);
			}

    		if (THREADS_PER_VECTOR >  8 && thread_lane < 8) {
				temp = sdata[0][threadIdx.x + 8];
				sdata[0][threadIdx.x] = temp0 = reduce(temp0, temp);
				temp = sdata[1][threadIdx.x + 8];
				sdata[1][threadIdx.x] = temp1 = reduce(temp1, temp);
				temp = sdata[2][threadIdx.x + 8];
				sdata[2][threadIdx.x] = temp2 = reduce(temp2, temp);
				temp = sdata[3][threadIdx.x + 8];
				sdata[3][threadIdx.x] = temp3 = reduce(temp3, temp);
			}

	    	if (THREADS_PER_VECTOR >  4 && thread_lane < 4) {
				temp = sdata[0][threadIdx.x + 4];
				sdata[0][threadIdx.x] = temp0 = reduce(temp0, temp);
				temp = sdata[1][threadIdx.x + 4];
				sdata[1][threadIdx.x] = temp1 = reduce(temp1, temp);
				temp = sdata[2][threadIdx.x + 4];
				sdata[2][threadIdx.x] = temp2 = reduce(temp2, temp);
				temp = sdata[3][threadIdx.x + 4];
				sdata[3][threadIdx.x] = temp3 = reduce(temp3, temp);
			}

	   		if (THREADS_PER_VECTOR >  2 && thread_lane < 2) {
				temp = sdata[0][threadIdx.x + 2];
				sdata[0][threadIdx.x] = temp0 = reduce(temp0, temp);
				temp = sdata[1][threadIdx.x + 2];
				sdata[1][threadIdx.x] = temp1 = reduce(temp1, temp);
				temp = sdata[2][threadIdx.x + 2];
				sdata[2][threadIdx.x] = temp2 = reduce(temp2, temp);
				temp = sdata[3][threadIdx.x + 2];
				sdata[3][threadIdx.x] = temp3 = reduce(temp3, temp);
			}
	   		
			if (THREADS_PER_VECTOR >  1 && thread_lane == 0) {
				temp = sdata[0][threadIdx.x + 1];
				sdata[0][threadIdx.x] = temp0 = reduce(temp0, temp);
				temp = sdata[1][threadIdx.x + 1];
				sdata[1][threadIdx.x] = temp1 = reduce(temp1, temp);
				temp = sdata[2][threadIdx.x + 1];
				sdata[2][threadIdx.x] = temp2 = reduce(temp2, temp);
				temp = sdata[3][threadIdx.x + 1];
				sdata[3][threadIdx.x] = temp3 = reduce(temp3, temp);
			}
			
			// if (thread_lane == 0) {
			// 	y[jj + rowIdx_start] = (ValueType)(sdata[jj][threadIdx.x]);
			// }
	   	// }
			
		if (thread_lane < rowIdx_end - rowIdx_start)
			y[rowIdx_start + thread_lane] = ValueType(sdata[thread_lane][threadIdx.x - thread_lane]);
		// if (thread_lane == 0) {
  		// 	for (IndexType row = 0; row < rowIdx_end - rowIdx_start; row++) {
		// 		y[row + rowIdx_start] = (ValueType)(sdata[row][threadIdx.x]);
		// 	}
		// }

 		// reduce local sums to row sum
//  		for (IndexType jj = 0; jj < rowIdx_end - rowIdx_start; jj ++) {
// 			sdata[jj][threadIdx.x] = sum[jj];
// 
//          	if (THREADS_PER_VECTOR > 16) {
//          	    temp = sdata[jj][threadIdx.x + 16];
//          	    sdata[jj][threadIdx.x] = sum[jj] = reduce(sum[jj], temp);
//          	}
//          	if (THREADS_PER_VECTOR >  8) {
//          	    temp = sdata[jj][threadIdx.x + 8];
//          	    sdata[jj][threadIdx.x] = sum[jj] = reduce(sum[jj], temp);
//          	}
//          	if (THREADS_PER_VECTOR >  4) {
//          	    temp = sdata[jj][threadIdx.x + 4];
//          	    sdata[jj][threadIdx.x] = sum[jj] = reduce(sum[jj], temp);
//          	}
//         	if (THREADS_PER_VECTOR >  2) {
//         	    temp = sdata[jj][threadIdx.x + 2];
//         	    sdata[jj][threadIdx.x] = sum[jj] = reduce(sum[jj], temp);
//         	}
//         	if (THREADS_PER_VECTOR >  1) {
//         	    temp = sdata[jj][threadIdx.x + 1];
//         	    sdata[jj][threadIdx.x] = sum[jj] = reduce(sum[jj], temp);
//         	}
// 			
// 			if (thread_lane == 0)
// 				y[jj + rowIdx_start] = (ValueType)(sdata[jj][threadIdx.x]);
//   		}
    }
}

template <unsigned int THREADS_PER_VECTOR,
         typename DerivedPolicy,
         typename MatrixType,
         typename VectorType1,
         typename VectorType2,
         typename UnaryFunction,
         typename BinaryFunction1,
         typename BinaryFunction2>
void __spmv_csr_vector(cuda::execution_policy<DerivedPolicy>& exec,
                       const MatrixType& A,
                       const VectorType1& x,
                       VectorType2& y,
                       UnaryFunction   initialize,
                       BinaryFunction1 combine,
                       BinaryFunction2 reduce)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    typedef typename MatrixType::row_offsets_array_type::const_iterator     RowIterator;
    typedef typename MatrixType::column_indices_array_type::const_iterator  ColumnIterator;
    typedef typename MatrixType::values_array_type::const_iterator          ValueIterator1;

    typedef typename VectorType1::const_iterator                            ValueIterator2;
    typedef typename VectorType2::iterator                                  ValueIterator3;

	const size_t THREADS_PER_BLOCK  = 512;
    const size_t VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;

    const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(
                                  spmv_csr_vector_kernel<RowIterator, ColumnIterator, ValueIterator1, ValueIterator2, ValueIterator3,
                                  UnaryFunction, BinaryFunction1, BinaryFunction2,
                                  VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, THREADS_PER_BLOCK, (size_t) 0);
    const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, DIVIDE_INTO(A.num_rows, VECTORS_PER_BLOCK));

    //std::cout << "Parameters Setting:" << "\n";
    //std::cout << "\t" << "VECTORS_PER_BLOCK = " << VECTORS_PER_BLOCK << "\n";
    //std::cout << "\t" << "THREADS_PER_VECTOR = " << THREADS_PER_VECTOR << "\n";
    //std::cout << "\t" << "NUM_BLOCKS = " << NUM_BLOCKS << "\n";
    //std::cout << "\t" << "THREADS_PER_BLOCK = " << THREADS_PER_BLOCK << "\n";
    
    cudaStream_t s = stream(thrust::detail::derived_cast(exec));

	cudaProfilerStart();
    spmv_csr_vector_kernel<RowIterator, ColumnIterator, ValueIterator1, ValueIterator2, ValueIterator3,
                           UnaryFunction, BinaryFunction1, BinaryFunction2,
                           VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, s>>>
                           (A.num_rows, A.row_offsets.begin(), A.column_indices.begin(), A.values.begin(), x.begin(), y.begin(),
                            initialize, combine, reduce);
	cudaProfilerStop();
}

template <typename DerivedPolicy,
          typename MatrixType,
          typename VectorType1,
          typename VectorType2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void multiply(cuda::execution_policy<DerivedPolicy>& exec,
              const MatrixType& A,
              const VectorType1& x,
              VectorType2& y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce,
              cusp::csr_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    typedef typename MatrixType::index_type IndexType;

    const IndexType nnz_per_row = A.num_entries / A.num_rows;
    //std::cout << "nnz_per_row=" << nnz_per_row << "\n";
    __spmv_csr_vector<16>(exec, A, x, y, initialize, combine, reduce);
    // if (nnz_per_row <=  2) {
    //     __spmv_csr_vector<2>(exec, A, x, y, initialize, combine, reduce);
    //     return;
    // }
    // if (nnz_per_row <=  4) {
    //     __spmv_csr_vector<4>(exec, A, x, y, initialize, combine, reduce);
    //     return;
    // }
    // if (nnz_per_row <=  8) {
    //     __spmv_csr_vector<8>(exec, A, x, y, initialize, combine, reduce);
    //     return;
    // }
    // if (nnz_per_row <= 16) {
    //     __spmv_csr_vector<16>(exec, A, x, y, initialize, combine, reduce);
    //     return;
    // }

    // __spmv_csr_vector<32>(exec, A, x, y, initialize, combine, reduce);
}

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp

