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
#include </home/gjh/gjh/BinarySpMV/bitCompress/RFCOC_onlyFold/bitCompress_cusp_v2.h>

#include <thrust/device_ptr.h>

#include <algorithm>

namespace cusp
{
namespace system
{
namespace cuda
{
namespace detail
{

////////////////////////////////////////////////////////////////////////
// CSR SpMV kernels based on a scalar model (one thread per row)
///////////////////////////////////////////////////////////////////////
//
// spmv_csr_scalar_device
//   Straightforward translation of standard CSR SpMV to CUDA
//   where each thread computes y[i] = A[i,:] * x
//   (the dot product of the i-th row of A with the x vector)
//
// spmv_csr_scalar_tex_device
//   Same as spmv_csr_scalar_device, except x is accessed via texture cache.
//

#define BLOCK_SIZE 32

template <typename IndexType,
          typename ValueType>
__global__ void
spmv_csr_scalar_kernel(const IndexType num_rows,
                       const IndexType * Ap,
                       const IndexType * Aj,
                       const ValueType * Ax,
                       const ValueType * x,
                       ValueType * y)
{ 
    const IndexType thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const IndexType grid_size = gridDim.x * blockDim.x;

	IndexType fold_bits = Ap[3 * (num_rows + 1)];
	IndexType foldgrain_max_value = Ap[3 * (num_rows +1) + 1];
	IndexType offset_bits = Ap[3 * (num_rows+1) + 2];
	IndexType offset_max_value = Ap[3 * (num_rows+1) + 3];
	// IndexType foldgrain_max_value = pow(2, fold_bits) - 1;
	// __shared__ ValueType sum[max_fold_num_rows * BLOCK_SIZE]; 
	__shared__ ValueType sum[64 * BLOCK_SIZE]; 

    for(IndexType row = thread_id; row < num_rows; row += grid_size)
    {
		const IndexType s_rowIdx = Ap[row];
		const IndexType e_rowIdx = Ap[row+1];
        const IndexType s_rowPtr_col = Ap[num_rows+1 + row];
        const IndexType e_rowPtr_col = Ap[num_rows+1 + row+1];
		IndexType s_rowPtr_val = Ap[(num_rows+1)*2 + row];
		IndexType base = 0;
		
        // ValueType sum[FOLDGRAIN] = {0}; // FOLDGRAIN = 4
		IndexType curColIdx, blockId, colIdx;
        for (IndexType jj = s_rowPtr_col; jj < e_rowPtr_col; jj++) {
            curColIdx = Aj[jj];
			IndexType sign = (curColIdx & 0xC0000000) >> 30;
			if (sign == 0) { // base
				blockId = (curColIdx & foldgrain_max_value);
				colIdx = (curColIdx >> fold_bits );
				// printf("1: thread %u: sum[%u] (%.6f) += Ax[%u] * x[%u] = %.6f * %.6f\n", thread_id, blockId*blockDim.x+threadIdx.x, sum[blockId*blockDim.x+threadIdx.x], s_rowPtr_val, colIdx, Ax[s_rowPtr_val], x[colIdx]);
				sum[blockId  * blockDim.x + threadIdx.x] += Ax[s_rowPtr_val++] * x[colIdx];
				base = colIdx;
			}
			else { // offset
				IndexType tmp_val = curColIdx;
				for (IndexType k = 0; k < sign; k++) { 
					blockId = tmp_val & foldgrain_max_value;
					tmp_val = tmp_val >> fold_bits;
					colIdx = base + (tmp_val & offset_max_value);
					 //printf("2: thread %u: sum[%u] (%.6f) += Ax[%u] * x[%u] = %.6f * %.6f\n", thread_id, blockId*blockDim.x+threadIdx.x, sum[blockId*blockDim.x+threadIdx.x], s_rowPtr_val, colIdx, Ax[s_rowPtr_val], x[colIdx]);
					sum[blockId * blockDim.x + threadIdx.x] += Ax[s_rowPtr_val++] * x[colIdx];
                    tmp_val = tmp_val >> offset_bits; // OFFSET = 8
				}
			}
        }

		for (IndexType i = 0; i < e_rowIdx - s_rowIdx; i++) {
			IndexType tmp_sum_index = i * BLOCK_SIZE + threadIdx.x;
         	y[s_rowIdx + i] = sum[tmp_sum_index];
			sum[tmp_sum_index] = 0.0;
		}
    }
}

template <typename DerivedPolicy,
         typename MatrixType,
         typename VectorType1,
         typename VectorType2,
         typename UnaryFunction,
         typename BinaryFunction1,
         typename BinaryFunction2>
void spmv_csr_scalar(cuda::execution_policy<DerivedPolicy>& exec,
                     MatrixType& A,
                     VectorType1& x,
                     VectorType2& y,
                     UnaryFunction   initialize,
                     BinaryFunction1 combine,
                     BinaryFunction2 reduce,
                     csr_format,
                     array1d_format,
                     array1d_format)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    const size_t block_row = A.num_rows;
    // const size_t BLOCK_SIZE = 256;
    const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_csr_scalar_kernel<IndexType, ValueType>, BLOCK_SIZE, (size_t) 0);
    const size_t NUM_BLOCKS = std::min(MAX_BLOCKS, DIVIDE_INTO(block_row, BLOCK_SIZE));

   //  std::cout << "BLOCK_SIZE=" << BLOCK_SIZE;
   //  std::cout << ", MAX_BLOCKS=" << MAX_BLOCKS;
   //  std::cout << ", NUM_BLOCKS=" << NUM_BLOCKS << "\n";
   cudaStream_t s = stream(thrust::detail::derived_cast(exec));

    spmv_csr_scalar_kernel<IndexType,ValueType> <<<NUM_BLOCKS, BLOCK_SIZE, 0, s>>>
    (block_row,
     thrust::raw_pointer_cast(&A.row_offsets[0]),
     thrust::raw_pointer_cast(&A.column_indices[0]),
     thrust::raw_pointer_cast(&A.values[0]),
     thrust::raw_pointer_cast(&x[0]),
     thrust::raw_pointer_cast(&y[0]));
}

template <typename MatrixType,
          typename VectorType1,
          typename VectorType2>
void spmv_csr_scalar(const MatrixType&  A,
                     const VectorType1& x,
                           VectorType2& y)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType::memory_space  System1;
    typedef typename VectorType1::memory_space System2;
    typedef typename VectorType2::memory_space System3;
    typedef typename MatrixType::value_type ValueType;

    System1 system1;
    System2 system2;
    System3 system3;

    cusp::constant_functor<ValueType> initialize(0);
    thrust::multiplies<ValueType> combine;
    thrust::plus<ValueType> reduce;

    spmv_csr_scalar(thrust::detail::derived_cast(
                      thrust::detail::strip_const(
                        select_system(system1,system2,system3))),
                    A, x, y,
                    initialize, combine, reduce,
                    csr_format(), array1d_format(), array1d_format());
}

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp

