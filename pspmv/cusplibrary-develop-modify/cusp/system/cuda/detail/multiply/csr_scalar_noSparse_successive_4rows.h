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

#define BLOCK_SIZE 256

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

	__shared__ ValueType sum[FOLDGRAIN * BLOCK_SIZE]; 

	IndexType base = 0;

    for(IndexType row = thread_id; row < num_rows; row += grid_size)
    {
        IndexType s_rowPtr = Ap[row];
        IndexType e_rowPtr = Ap[row+1];
        IndexType s_valIdx = Ap[num_rows+1 + row];

        for (IndexType jj = s_rowPtr; jj < e_rowPtr; jj++) {
            IndexType curColIdx = Aj[jj];
            IndexType sign = (curColIdx & 0xC0000000) >> 30;	// get the highest two bits
            if (sign == 0) {	// base
				IndexType blockId = (curColIdx & FOLDGRAIN_MAX_VALUE);
				IndexType colIdx = (curColIdx >> FOLDGRAIN);
				
				for (IndexType i = 0; i < FOLDGRAIN; i++) {
					if (blockId & 1 == 1) 
						sum[i  * blockDim.x + threadIdx.x] += Ax[s_valIdx++] * x[colIdx];
					blockId = blockId >> 1;
				}
			// 	IndexType each_blockId = (blockId & 1);
			// 	if (each_blockId == 1) {
			// 		sum[0] += Ax[s_valIdx++] * x[colIdx];
			// 	}
			// 	each_blockId = (blockId & 2);
			// 	if (each_blockId == 2) {
			// 		sum[1] += Ax[s_valIdx++] * x[colIdx];
			// 	}
			// 	each_blockId = (blockId & 4);
			// 	if (each_blockId == 4) {
			// 		sum[2] += Ax[s_valIdx++] * x[colIdx];
			// 	}
			// 	each_blockId = (blockId & 8);
			// 	if (each_blockId == 8) {
			// 		sum[3] += Ax[s_valIdx++] * x[colIdx];
			// 	}

				base = colIdx;
			}
			else {	// offset, sign = 1 or 2 or 3
				IndexType tmp_val = curColIdx;
				IndexType blockId, colIdx;

#pragma unroll
				for (IndexType i = 0; i < sign; i++) {
					blockId = tmp_val & SHIFT_BITS_MAX_VALUE;
					tmp_val = tmp_val >> SHIFT_BITS;
					colIdx = base + (tmp_val & OFFSET_BITS_POWER);
					sum[blockId * blockDim.x + threadIdx.x] += Ax[s_valIdx++] * x[colIdx];
					tmp_val = tmp_val >> OFFSET_BITS;

				}
			// 	if (sign == 3) {
			// 		blockId = tmp_val & 3;// 3 = (int)(pow(2,BLOCKGRAIN_BITS)-1
			// 		tmp_val = tmp_val >> 2;
			// 		colIdx = base + (tmp_val & 0xFF);// 0xFF = (int)(pow(2,OFFSET) - 1
			// 		sum[blockId] += Ax[s_valIdx++] * x[colIdx];
            //         tmp_val = tmp_val >> 8; // OFFSET = 8

			// 		blockId = tmp_val & 3;// 3 = (int)(pow(2,BLOCKGRAIN_BITS)-1
			// 		tmp_val = tmp_val >> 2;
			// 		colIdx = base + (tmp_val & 0xFF);// 0xFF = (int)(pow(2,OFFSET) - 1
			// 		sum[blockId] += Ax[s_valIdx++] * x[colIdx];
            //         tmp_val = tmp_val >> 8; // OFFSET = 8

			// 		blockId = tmp_val & 3;// 3 = (int)(pow(2,BLOCKGRAIN_BITS)-1
			// 		tmp_val = tmp_val >> 2;
			// 		colIdx = base + (tmp_val & 0xFF);// 0xFF = (int)(pow(2,OFFSET) - 1
			// 		sum[blockId] += Ax[s_valIdx++] * x[colIdx];
			// 	}
			// 	else if (sign == 2) {
			// 		blockId = tmp_val & 3;// 3 = (int)(pow(2,BLOCKGRAIN_BITS)-1
			// 		tmp_val = tmp_val >> 2;
			// 		colIdx = base + (tmp_val & 0xFF);// 0xFF = (int)(pow(2,OFFSET) - 1
			// 		sum[blockId] += Ax[s_valIdx++] * x[colIdx];
            //         tmp_val = tmp_val >> 8; // OFFSET = 8

			// 		blockId = tmp_val & 3;// 3 = (int)(pow(2,BLOCKGRAIN_BITS)-1
			// 		tmp_val = tmp_val >> 2;
			// 		colIdx = base + (tmp_val & 0xFF);// 0xFF = (int)(pow(2,OFFSET) - 1
			// 		sum[blockId] += Ax[s_valIdx++] * x[colIdx];
			// 	}
			// 	else { // sign == 1
			// 		blockId = tmp_val & 3;// 3 = (int)(pow(2,BLOCKGRAIN_BITS)-1
			// 		tmp_val = tmp_val >> 2;
			// 		colIdx = base + (tmp_val & 0xFF);// 0xFF = (int)(pow(2,OFFSET) - 1
			// 		sum[blockId] += Ax[s_valIdx++] * x[colIdx];
            //         tmp_val = tmp_val >> 8; // OFFSET = 8
			// 	}
			}
        }

		IndexType tmp = row * FOLDGRAIN;
#pragma unroll
		for (IndexType  i = 0; i < FOLDGRAIN; i++) {
			y[tmp + i] = sum[i * blockDim.x  +  threadIdx.x];
			sum[i * blockDim.x + threadIdx.x] = 0;
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

    const size_t block_row = (A.num_rows-1)  / 2;
    // const size_t BLOCK_SIZE = 256;
    const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_csr_scalar_kernel<IndexType, ValueType>, BLOCK_SIZE, (size_t) 0);
    const size_t NUM_BLOCKS = std::min(MAX_BLOCKS, DIVIDE_INTO(block_row, BLOCK_SIZE));

    // std::cout << "BLOCK_SIZE=" << BLOCK_SIZE;
    // std::cout << ", MAX_BLOCKS=" << MAX_BLOCKS;
    // std::cout << ", NUM_BLOCKS=" << NUM_BLOCKS << "\n";
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
