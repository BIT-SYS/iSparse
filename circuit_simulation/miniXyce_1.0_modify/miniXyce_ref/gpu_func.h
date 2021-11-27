#ifndef _GPU_FUNC_H_
#define _GPU_FUNC_H_

#include <cuda_runtime.h>
#include "cusparse.h"
// #include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <cusp/system/cuda/arch.h>
#include "cuda_profiler_api.h"

#include "mX_sparse_matrix.h"

typedef std::vector<int> int_vector_host;
typedef std::vector<double> double_vector_host;
typedef thrust::device_vector<int> int_vector_device;
typedef thrust::device_vector<double> double_vector_device;

// using mX_matrix_utils::distributed_sparse_matrix;
// void matrix_transfer_CPUToGPU(distributed_sparse_matrix* A, distributed_sparse_matrix*A_dev);
// void vector_transfer_CPUToGPU(std::vector<double> x, std::vector<double>x_dev);
void my_SpMV(int M, int N, int nnz, int_vector_device& csrRowPtr, int_vector_device& csrColIdx, double_vector_device& csrVal, double_vector_device& x, double_vector_device& y);
void CSR_coop_spmv_prepare0(int M, int TPV, int_vector_device& RowPtr, int_vector_device &ColIdx, double_vector_device& Val, double_vector_device& x, double_vector_device& y);
void matrix_to_three_vector(int &M, int &N, int &nnz, mX_matrix_utils::distributed_sparse_matrix* A, int_vector_device& csrRowPtr, int_vector_device& csrColIdx, double_vector_device& csrVal);
void my_gmres(int M, int N, int nnz, int_vector_device &csrRowPtr, int_vector_device &csrColIdx, 
                            double_vector_device &csrVal, double_vector_host &b, double_vector_host &x0, 
                            double &tol, double &err, int k, double_vector_host &x, int &iters, int &restarts);


inline void checkcuda(cudaError_t result)
{
	if (result != cudaSuccess)
	{
		printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		printf("hello");
	}
}

inline void checkcusparse(cusparseStatus_t result)
{
	if(result != CUSPARSE_STATUS_SUCCESS){
		printf("CUSPARSE Error, error_code =  %d\n", result);
	}
}

#endif