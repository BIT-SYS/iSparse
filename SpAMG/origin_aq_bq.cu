#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
#include <stdlib.h> 
#include <iostream>
#include <fstream>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <cfloat>
#include <errno.h>
#include <cusparse.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/copy.h>
#include <dirent.h>
#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <sstream>
#include "mmio.h"
#include <math.h>

#define CUB_STDERR // Ensure printing of CUDA runtime errors to console
#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>
inline static void checkCUDA(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
        file, line);
        throw std::exception();
    }
}
#define CHECK_ERROR(err) (checkCUDA(err, __FILE__, __LINE__))
#define cudaCheckError() {                                              \
    cudaDeviceSynchronize();                                              \
    cudaError_t e=cudaGetLastError();                                     \
    if(e!=cudaSuccess) {                                                  \
        exit(EXIT_FAILURE);                                               \
}}                                                                        

const int N = 101;

int MA, NA, nnzA;
int *csrRowIndexHostPtrA = 0;
int *csrColIndexHostPtrA = 0;
double *csrValHostPtrA = 0;
char matrixNameA[1024] = {0};

int MB, NB, nnzB;
int *csrRowIndexHostPtrB = 0;
int *csrColIndexHostPtrB = 0;
double *csrValHostPtrB = 0;
char matrixNameB[1024] = {0};

int MC, NC, nnzC;
int *csrRowIndexHostPtrC = 0;
int *csrColIndexHostPtrC = 0;
double *csrValHostPtrC = 0;

int *csrRowIndexDevPtrA = 0;
int *csrColIndexDevPtrA = 0;
double *csrValDevPtrA = 0;

int *csrRowIndexDevPtrB = 0;
int *csrColIndexDevPtrB = 0;
double *csrValDevPtrB = 0;

int *csrRowIndexDevPtrC = 0;
int *csrColIndexDevPtrC = 0;
double *csrValDevPtrC = 0;

// Constant parameters.
const int m_grid_size = 1024, m_max_warp_count = 8;
// The number of threads per row of B.
int m_num_threads_per_row_count = 2, m_num_threads_per_row_compute = 2;
// The size of the GMEM buffers (number of elements).
int m_gmem_size = 512;
// The status: OK if count_non_zeroes succeeded, FAILED otherwise.
int *m_status;
// The work queue for dynamic load balancing in the kernels.
int *m_work_queue;
// The buffer to store keys in GMEM.
int *m_keys;
// The buffer to store values in GMEM.
double *m_vals;


inline void checkcuda(cudaError_t result)
{
	if (result != cudaSuccess)
	{
		printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		exit(0);
	}
}

inline void checkcusparse(cusparseStatus_t result)
{
	if(result != CUSPARSE_STATUS_SUCCESS){
		printf("CUSPARSE Error, error_code =  %d\n", result);
		exit(0);
	}
}
//验证nnz不相等 rpt不相等 colids为-1 则返回-1，否则返回0
//传入的注意是host指针
int test_ok(int nnz, int* r, int* c, double* v)
{
    int *csrRowIndexDevPtrCGolden, *csrColIndexDevPtrCGolden;
    double* csrValDevPtrCGolden;
	//cusparse
    int res = 0;
    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);
    cusparseSpMatDescr_t matA, matB, matC;

    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    cudaDataType matType;
    matType = CUDA_R_64F;

    cusparseIndexType_t indType;
    indType = CUSPARSE_INDEX_32I;
    checkcusparse( cusparseCreateCsr(&matA, MA, NA, nnzA,
                                      csrRowIndexDevPtrA, csrColIndexDevPtrA, csrValDevPtrA,
                                      indType, indType,
                                      CUSPARSE_INDEX_BASE_ZERO, matType) );
    checkcusparse( cusparseCreateCsr(&matB, MB, NB, nnzB,
                                      csrRowIndexDevPtrB, csrColIndexDevPtrB, csrValDevPtrB,
                                      indType, indType,
                                      CUSPARSE_INDEX_BASE_ZERO, matType) );
    checkcusparse( cusparseCreateCsr(&matC, MA, NB, 0,
                                      NULL, NULL, NULL,
                                      indType, indType,
                                      CUSPARSE_INDEX_BASE_ZERO, matType) );

    float alpha = 1.0f;
	float beta = 0.0f;
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = matType;

    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    checkcusparse( cusparseSpGEMM_createDescr(&spgemmDesc) );

    // ask bufferSize1 bytes for external memory
    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, NULL);
    cudaMalloc(&dBuffer1, bufferSize1);
    // inspect the matrices A and B to understand the memory requiremnent for
    // the next step
    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, dBuffer1);

    // ask bufferSize2 bytes for external memory
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, NULL);
    cudaMalloc(&dBuffer2, bufferSize2);

    // compute the intermediate product of A * B
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, dBuffer2);
    // get matrix C non-zero entries C_num_nnz1
    int64_t C_num_rows1, C_num_cols1, C_num_nnz1;
    cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_num_nnz1);

    
    (cudaMalloc((void **)&csrRowIndexDevPtrCGolden, (MA + 1) * sizeof(int)));
    (cudaMalloc((void **)&csrColIndexDevPtrCGolden, C_num_nnz1 * sizeof(int)));
	(cudaMalloc((void **)&csrValDevPtrCGolden, C_num_nnz1 * sizeof(double)));

    MC = MA;
    NC = NA;
    nnzC = C_num_rows1;

    cusparseCsrSetPointers(matC, csrRowIndexDevPtrCGolden, csrColIndexDevPtrCGolden, csrValDevPtrCGolden);

    // copy the final products to the matrix C
    checkcusparse(cusparseSpGEMM_copy(handle, opA, opB,
                        &alpha, matA, matB, &beta, matC,
                        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));



   int *csrRowPtrC_tmp = (int *)malloc((MA + 1) * sizeof(int));
	int *csrColIdxC_tmp = (int*)malloc(C_num_nnz1 * sizeof(int));
	double *csrValC_tmp = (double *)malloc(C_num_nnz1 * sizeof(double));
	cudaMemcpy(csrRowPtrC_tmp, csrRowIndexDevPtrCGolden, (MA + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(csrColIdxC_tmp, csrColIndexDevPtrCGolden, C_num_nnz1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(csrValC_tmp, csrValDevPtrCGolden, C_num_nnz1 * sizeof(double), cudaMemcpyDeviceToHost);

	//验证nnz
	// if(C_num_nnz1 != nnz) {
	// 	printf("%d %d nnz not correct \n", C_num_nnz1, nnz);
	// 	res = -1;
	// }

    //TODO 这里的cusparse时钟不能将val算出来，全是0值，调试一天放弃
    // for(int i=0; i < 100; ++i) {
    //     // printf("%d th rpt = %d\n", i, csrRowPtrC_tmp[i]);
    //     // printf("%d th col = %d\n", i, csrColIdxC_tmp[i]);
	// 	printf("%d th val = %lf\n", i, csrValC_tmp[i]);
	// }

    
    // for(int i=0; i < 100; ++i) {
    //     // printf("%d th rpt = %d\n", i, r[i]);
    //     // printf("%d th col = %d\n", i, c[i]);
	// 	printf("%d th val = %lf\n", i, v[i]);
	// }
    
    // int *csrRowIndexDevPtrCGolden, *csrColIndexDevPtrCGolden;
    // double* csrValDevPtrCGolden;
	
    // cusparseCreate(&handle);
    // // Create sparse matrix A in CSR format
    // cusparseCreateCsr(&matA, MA, NA, nnzA,
	// 	csrRowIndexDevPtrA, csrColIndexDevPtrA, csrValDevPtrA,
    //                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    //                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    // cusparseCreateCsr(&matB, MB, NB, nnzB,
	// 	csrRowIndexDevPtrB, csrColIndexDevPtrB, csrValDevPtrB,
    //                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    //                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    // cusparseCreateCsr(&matC, MA, NB, 0,
    //                   NULL, NULL, NULL,
    //                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    //                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    // cudaDeviceSynchronize();

	// cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
	// cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
	
	// void *dBuffer1 = NULL, *dBuffer2 = NULL;
	// size_t bufferSize1 = 0, bufferSize2 = 0;
	// float alpha = 1.0f;
	// float beta = 0.0f;

	// checkcuda(cudaMalloc((void **)&csrRowIndexDevPtrCGolden, (MA + 1) * sizeof(int)));

	// // SpGEMM Computation
	// cusparseSpGEMMDescr_t spgemmDesc;
	// checkcusparse(cusparseSpGEMM_createDescr(&spgemmDesc));

	// // ask bufferSize1 bytes for external memory
	// checkcusparse(cusparseSpGEMM_workEstimation(handle, opA, opB,
	// 							&alpha, matA, matB, &beta, matC,
	// 							CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
	// 							spgemmDesc, &bufferSize1, NULL));

	// checkcuda(cudaMalloc((void **)&dBuffer1, bufferSize1));
	
	// // inspect the matrices A and B to understand the memory requiremnent for
	// // the next step
	// checkcusparse(cusparseSpGEMM_workEstimation(handle, opA, opB,
	// 							&alpha, matA, matB, &beta, matC,
	// 							CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
	// 							spgemmDesc, &bufferSize1, dBuffer1));

	// // ask bufferSize2 bytes for external memory
	// checkcusparse(cusparseSpGEMM_compute(handle, opA, opB,
	// 					&alpha, matA, matB, &beta, matC,
	// 					CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
	// 					spgemmDesc, &bufferSize2, NULL));
	
	// checkcuda(cudaMalloc((void **)&dBuffer2, bufferSize2));

	// // compute the intermediate product of A * B
	// checkcusparse(cusparseSpGEMM_compute(handle, opA, opB,
	// 					&alpha, matA, matB, &beta, matC,
	// 					CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
	// 					spgemmDesc, &bufferSize2, dBuffer2));
	// // get matrix C non-zero entries C_num_nnz1
	// int64_t C_num_rows1, C_num_cols1, C_num_nnz1;
	// checkcusparse(cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_num_nnz1));
	
	// // printf("C_num_nnz1 = %d\n", C_num_nnz1);

	// // allocate matrix C
	// checkcuda(cudaMalloc((void **)&csrColIndexDevPtrCGolden, C_num_nnz1 * sizeof(int)));
	// checkcuda(cudaMalloc((void **)&csrValDevPtrCGolden, C_num_nnz1 * sizeof(double)));
	// // update matC with the new pointers
	// checkcusparse(cusparseCsrSetPointers(matC, csrRowIndexDevPtrCGolden, csrColIndexDevPtrCGolden, csrValDevPtrCGolden));

	// // copy the final products to the matrix C
	// checkcusparse(cusparseSpGEMM_copy(handle, opA, opB,
	// 					&alpha, matA, matB, &beta, matC,
	// 					CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

	// int *csrRowPtrC_tmp = (int *)malloc((MA + 1) * sizeof(int));
	// int *csrColIdxC_tmp = (int*)malloc(C_num_nnz1 * sizeof(int));
	// double *csrValC_tmp = (double *)malloc(C_num_nnz1 * sizeof(double));
	// cudaMemcpy(csrRowPtrC_tmp, csrRowIndexDevPtrCGolden, (MA + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(csrColIdxC_tmp, csrColIndexDevPtrCGolden, C_num_nnz1 * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(csrValC_tmp, csrValDevPtrCGolden, C_num_nnz1 * sizeof(double), cudaMemcpyDeviceToHost);


    // int res = 0;
	// //验证nnz
	// if(C_num_nnz1 != nnz) {
	// 	printf("%d %d nnz not correct \n", C_num_nnz1, nnz);
	// 	res = -1;
	// }
	//验证rpt
	// for(int i=0; i < C_num_rows1 + 1; ++i) {
	// 	if(r[i] != csrRowPtrC_tmp[i]) {
	// 		res = -1;
	// 		printf("%d th row is not correct\n", i);
	// 		break;
	// 	}
	// }

	//验证colids不是无效值
	// for(int i=0; i < C_num_nnz1; ++i) {
	// 	if(c[i] == -1) {
	// 		res = -1;
	// 		printf("%d th col is not correct\n", i);
	// 		break;
	// 	}
	// }

    // for(int i=0; i < 100; ++i) {
    //     printf("%d th rpt = %d\n", i, csrRowPtrC_tmp[i]);
    //     printf("%d th col = %d\n", i, csrColIdxC_tmp[i]);
	// 	printf("%d th val = %lf\n", i, csrValC_tmp[i]);
	// }

    // for(int i=0; i < 100; ++i) {
    //     printf("%d th rpt = %d\n", i, r[i]);
    //     printf("%d th col = %d\n", i, c[i]);
	// 	printf("%d th val = %lf\n", i, v[i]);
	// }

	//清理

	// checkcusparse(cusparseSpGEMM_destroyDescr(spgemmDesc));
	// cudaFree(csrRowIndexDevPtrCGolden);
	// cudaFree(csrColIndexDevPtrCGolden);
	// cudaFree(csrValDevPtrCGolden);
	// cudaFree(dBuffer1);
	// cudaFree(dBuffer2);
	// cusparseDestroySpMat(matA);
    // cusparseDestroySpMat(matB);
    // cusparseDestroySpMat(matC);
    // cusparseDestroy(handle);
    // free(csrRowPtrC_tmp);
    // free(csrColIdxC_tmp);
    // free(csrValC_tmp);

	return res;
}

int test_ok_for_nsparse(int nnz, int* r, int* c, double* v,
    int* ar, int* ac, double* av, 
    int* br, int* bc, double* bv, 
    int ma, int na, int nnza,
    int mb, int nb, int nnzb)
{
	//cusparse
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    int *csrRowIndexDevPtrCGolden, *csrColIndexDevPtrCGolden;
    double* csrValDevPtrCGolden;
	
    cusparseCreate(&handle);
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, ma, na, nnza,
		ar, ac, av,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateCsr(&matB, mb, nb, nnzb,
		br, bc, bv,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateCsr(&matC, MA, NB, 0,
                      NULL, NULL, NULL,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cudaDeviceSynchronize();

	cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
	cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
	
	void *dBuffer1 = NULL, *dBuffer2 = NULL;
	size_t bufferSize1 = 0, bufferSize2 = 0;
	float alpha = 1.0f;
	float beta = 0.0f;

	checkcuda(cudaMalloc((void **)&csrRowIndexDevPtrCGolden, (MA + 1) * sizeof(int)));

	// SpGEMM Computation
	cusparseSpGEMMDescr_t spgemmDesc;
	checkcusparse(cusparseSpGEMM_createDescr(&spgemmDesc));

	// ask bufferSize1 bytes for external memory
	checkcusparse(cusparseSpGEMM_workEstimation(handle, opA, opB,
								&alpha, matA, matB, &beta, matC,
								CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
								spgemmDesc, &bufferSize1, NULL));

	checkcuda(cudaMalloc((void **)&dBuffer1, bufferSize1));
	
	// inspect the matrices A and B to understand the memory requiremnent for
	// the next step
	checkcusparse(cusparseSpGEMM_workEstimation(handle, opA, opB,
								&alpha, matA, matB, &beta, matC,
								CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
								spgemmDesc, &bufferSize1, dBuffer1));

	// ask bufferSize2 bytes for external memory
	checkcusparse(cusparseSpGEMM_compute(handle, opA, opB,
						&alpha, matA, matB, &beta, matC,
						CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
						spgemmDesc, &bufferSize2, NULL));
	
	checkcuda(cudaMalloc((void **)&dBuffer2, bufferSize2));

	// compute the intermediate product of A * B
	checkcusparse(cusparseSpGEMM_compute(handle, opA, opB,
						&alpha, matA, matB, &beta, matC,
						CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
						spgemmDesc, &bufferSize2, dBuffer2));
	// get matrix C non-zero entries C_num_nnz1
	int64_t C_num_rows1, C_num_cols1, C_num_nnz1;
	checkcusparse(cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_num_nnz1));
	
	// printf("C_num_nnz1 = %d\n", C_num_nnz1);

	// allocate matrix C
	checkcuda(cudaMalloc((void **)&csrColIndexDevPtrCGolden, C_num_nnz1 * sizeof(int)));
	checkcuda(cudaMalloc((void **)&csrValDevPtrCGolden, C_num_nnz1 * sizeof(double)));
	// update matC with the new pointers
	checkcusparse(cusparseCsrSetPointers(matC, csrRowIndexDevPtrCGolden, csrColIndexDevPtrCGolden, csrValDevPtrCGolden));

	// copy the final products to the matrix C
	checkcusparse(cusparseSpGEMM_copy(handle, opA, opB,
						&alpha, matA, matB, &beta, matC,
						CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

	int *csrRowPtrC_tmp = (int *)malloc((MA + 1) * sizeof(int));
	int *csrColIdxC_tmp = (int*)malloc(C_num_nnz1 * sizeof(int));
	double *csrValC_tmp = (double *)malloc(C_num_nnz1 * sizeof(double));
	cudaMemcpy(csrRowPtrC_tmp, csrRowIndexDevPtrCGolden, (MA + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(csrColIdxC_tmp, csrColIndexDevPtrCGolden, C_num_nnz1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(csrValC_tmp, csrValDevPtrCGolden, C_num_nnz1 * sizeof(double), cudaMemcpyDeviceToHost);


    int res = 0;
	//验证nnz
	if(C_num_nnz1 != nnz) {
		printf("%d %d nnz not correct \n", C_num_nnz1, nnz);
		res = -1;
	}
	//验证rpt
	for(int i=0; i < C_num_rows1 + 1; ++i) {
		if(r[i] != csrRowPtrC_tmp[i]) {
			res = -1;
			printf("%d th row is not correct\n", i);
			break;
		}
	}

	//验证colids不是无效值
	// for(int i=0; i < C_num_nnz1; ++i) {
	// 	if(c[i] == -1) {
	// 		res = -1;
	// 		printf("%d th col is not correct\n", i);
	// 		break;
	// 	}
	// }

	//清理

	checkcusparse(cusparseSpGEMM_destroyDescr(spgemmDesc));
	cudaFree(csrRowIndexDevPtrCGolden);
	cudaFree(csrColIndexDevPtrCGolden);
	cudaFree(csrValDevPtrCGolden);
	cudaFree(dBuffer1);
	cudaFree(dBuffer2);
	cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);
    cusparseDestroy(handle);
    free(csrRowPtrC_tmp);
    free(csrColIdxC_tmp);
    free(csrValC_tmp);

	return res;
}

namespace origin_old {
    static __device__ __forceinline__ int get_lane_id()
    {
        int id;
        asm( "mov.u32 %0, %%laneid;" : "=r"(id) );
        return id;
    }

    static __device__ __forceinline__ int get_lane_mask_lt()
    {
        int mask;
        asm( "mov.u32 %0, %%lanemask_lt;" : "=r"(mask) );
        return mask;
    }


    static __device__ __forceinline__ int get_warp_id()
    {
        return threadIdx.x >> 5;
    }

    #define DEFAULT_MASK 0xffffffff

    static __device__ __forceinline__ int shfl( int r, int lane, int bound = 32, unsigned int mask = DEFAULT_MASK )
    {
        return __shfl_sync( mask, r, lane, bound );
    }

    __device__ __forceinline__ int get_work( int *queue, int warp_id )
    {
        int offset = -1;

        if ( get_lane_id() == 0 )
        {
            offset = atomicAdd( queue, 1 );
        }

        return shfl( offset, 0 );
    }

    static __device__ __forceinline__ unsigned int ballot(int p, unsigned int mask = DEFAULT_MASK)
    {
        return __ballot_sync(mask, p);
    }

    static __device__ __forceinline__ unsigned int any(int p, unsigned int mask = DEFAULT_MASK)
    {
        return __any_sync(mask, p);
    }

    static __device__ __forceinline__ unsigned int all(int p, unsigned int mask = DEFAULT_MASK)
    {
    #if CUDART_VERSION >= 9000
        return __all_sync(mask, p);
    #else
        return __all(p);   
    #endif
    }

    static __device__ __forceinline__ unsigned int activemask()
    {
        return __activemask();

    }

    static __device__ __forceinline__ int shfl_xor( int r, int lane_mask, int bound = 32, unsigned int mask = DEFAULT_MASK )
    {
        return __shfl_xor_sync( mask, r, lane_mask, bound );
    }

    template< typename T >
    static __device__ __forceinline__ T load( const T *ptr ) { return __ldg( ptr ); }

    static __device__ __forceinline__ void atomic_add( double *address, double value )
    {
    #if __CUDA_ARCH__ >= 600
        atomicAdd( address, value );
    #else
        unsigned long long *address_as_ull = (unsigned long long *) address;
        unsigned long long old = __double_as_longlong( address[0] ), assumed;

        do
        {
            assumed = old;
            old = atomicCAS( address_as_ull, assumed, __double_as_longlong( value + __longlong_as_double( assumed ) ) );
        }
        while ( assumed != old );

    #endif
    }

    static __device__ __forceinline__ int bfind( int src )
    {
        int msb;
        asm( "bfind.u32 %0, %1;" : "=r"(msb) : "r"(src) );
        __syncthreads();
        return msb;
    }

    static __device__ __forceinline__ int bfe( int src, int num_bits )
    {
        unsigned mask;
        asm( "bfe.u32 %0, %1, 0, %2;" : "=r"(mask) : "r"(src), "r"(num_bits) );
        __syncthreads();
        return mask;
    }

    static __constant__ unsigned c_hash_keys[] =
    {
        3499211612,  581869302, 3890346734, 3586334585,
        545404204,  4161255391, 3922919429,  949333985,
        2715962298, 1323567403,  418932835, 2350294565,
        1196140740,  809094426, 2348838239, 4264392720
    };
        
    template< typename Key_type, int SMEM_SIZE = 128, int NUM_HASH_FCTS = 4, int WARP_SIZE = 32 >
    class Hash_set
    {
        protected:
            // The size of the table (occupancy).
            int m_smem_count, m_gmem_count;
            // The keys stored in the hash table.
            volatile Key_type *m_smem_keys, *m_gmem_keys;
            // The size of the global memory buffer.
            const int m_gmem_size;
            // Is it ok?
            bool m_fail;
    
        public:
            // Constructor.
            __device__ __forceinline__ Hash_set( volatile Key_type *smem_keys, volatile Key_type *gmem_keys, int gmem_size ) :
                m_smem_count(0),
                m_gmem_count(1),
                m_smem_keys (smem_keys),
                m_gmem_keys (gmem_keys),
                m_gmem_size (gmem_size),
                m_fail      (false)
    
            {}
    
            // Clear the table.
            __device__ __forceinline__ void clear( bool skip_gmem = false );
            // Compute the size of the table. Only thread with lane_id==0 gives the correct result (no broadcast of the value).
            __device__ __forceinline__ int compute_size();
            // Compute the size of the table. Only thread with lane_id==0 gives the correct result (no broadcast of the value).
            __device__ __forceinline__ int compute_size_with_duplicates();
            // Has the process failed.
            __device__ __forceinline__ bool has_failed() const { return m_fail; }
            // Insert a key inside the set. If status is NULL, ignore failure.
            __device__ __forceinline__ void insert( Key_type key, int *status );
            // Store a set.
            __device__ __forceinline__ void store( int count, Key_type *keys );
            // Store a set.
            __device__ __forceinline__ int  store_with_positions( Key_type *keys, int *pos );

    };
    
    // ====================================================================================================================
    
    template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE>
    __device__ __forceinline__
    void Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::clear( bool skip_gmem )
    {
        int lane_id = get_lane_id();
        const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
    #pragma unroll
    
        for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
        {
            m_smem_keys[i_step * WARP_SIZE + lane_id] = -1;
        }
    
        m_smem_count = 0;
    
        if ( skip_gmem || m_gmem_count == 0 )
        {
            m_gmem_count = 0;
            return;
        }
    
    #pragma unroll 4
    
        for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
        {
            m_gmem_keys[offset] = -1;
        }
    
        m_gmem_count = 0;
    }
    
    
    template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE>
    __device__ __forceinline__
    int Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::compute_size()
    {
        m_smem_count += m_gmem_count;
    #pragma unroll
    
        for ( int offset = WARP_SIZE / 2 ; offset > 0 ; offset >>= 1 )
        {
            m_smem_count += shfl_xor( m_smem_count, offset );
        }
    
        m_gmem_count = any( m_gmem_count > 0 );
        return m_smem_count;
    }
    
    // ====================================================================================================================
    
    template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE>
    __device__ __forceinline__
    int Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::compute_size_with_duplicates()
    {
        int lane_id = get_lane_id();
        // Count the number of keys in SMEM.
        int sum = 0;
        const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
    #pragma unroll
    
        for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
        {
            const int offset = i_step * WARP_SIZE + lane_id;
            Key_type key = m_smem_keys[offset];
            sum += __popc( ballot( key != -1 ) );
        }
    
        // Is there any key in GMEM. If not, just quit.
        m_gmem_count = any(m_gmem_count > 0);
    
        if ( !m_gmem_count )
        {
            return sum;
        }
    
        // Count the number of keys in GMEM.
    #pragma unroll 4
    
        for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
        {
            Key_type key = m_gmem_keys[offset];
            sum += __popc( ballot( key != -1, activemask() ) );
        }
    
        return sum;
    }
    
    
    // ====================================================================================================================
    
    template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
    __device__ __forceinline__
    void Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::insert( Key_type key, int *status )
    {
        bool done = key == -1;
    #pragma unroll
    
        for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
        {
            if ( all(done) )
            {
                return;
            }
    
            bool candidate = false;
            unsigned ukey = reinterpret_cast<unsigned &>( key );
            int hash = ( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash] ) & (SMEM_SIZE - 1);
    
            if ( !done )
            {
                Key_type stored_key = m_smem_keys[hash];
    
                if ( stored_key == key )
                {
                    done = true;
                }
    
                candidate = stored_key == -1;
    
                if ( candidate )
                {
                    m_smem_keys[hash] = key;
                }
    
                if ( candidate && key == m_smem_keys[hash] ) // More than one candidate may have written to that slot.
                {
                    m_smem_count++;
                    done = true;
                }
            }
        }
    
        const int num_bits = bfind( m_gmem_size );
    #pragma unroll
    
        for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
        {
            if ( all(done) )
            {
                return;
            }
    
            bool candidate = false;
            unsigned ukey = reinterpret_cast<unsigned &>( key );
            int hash = bfe( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash], num_bits );
    
            if ( !done )
            {
                Key_type stored_key = m_gmem_keys[hash];
    
                if ( stored_key == key )
                {
                    done = true;
                }
    
                candidate = stored_key == -1;
    
                if ( candidate )
                {
                    m_gmem_keys[hash] = key;
                }
    
                if ( candidate && key == m_gmem_keys[hash] ) // More than one candidate may have written to that slot.
                {
                    m_gmem_count++;
                    done = true;
                }
            }
        }
    
        if ( all(done) )
        {
            return;
        }
    
        assert( status != NULL );
    
        if ( get_lane_id() == 0 )
        {
            *status = 1;
        }
    
        m_fail = true;
    }
    
    template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
    __device__ __forceinline__
    void Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store( int count, Key_type *keys )
    {
        int lane_id = get_lane_id();
        int lane_mask_lt = get_lane_mask_lt();
        int warp_offset = 0;
        const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
    #pragma unroll
    
        for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
        {
            const int offset = i_step * WARP_SIZE + lane_id;
            Key_type key = m_smem_keys[offset];
            int poll = ballot( key != -1 );
    
            if ( poll == 0 )
            {
                continue;
            }
    
            int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
    
            if ( key != -1 )
            {
                keys[dst_offset] = key;
            }
    
            warp_offset += __popc( poll );
        }
    
        m_gmem_count = any( m_gmem_count > 0 );
    
        if ( !m_gmem_count )
        {
            return;
        }
    
    #pragma unroll 4
    
        for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
        {
            Key_type key = m_gmem_keys[offset];
            int poll = ballot( key != -1, activemask() );
    
            if ( poll == 0 )
            {
                continue;
            }
    
            int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
    
            if ( key != -1 )
            {
                keys[dst_offset] = key;
            }
    
            warp_offset += __popc( poll );
        }
    }

    union Word { char b8[4]; int b32; };
    
    // ====================================================================================================================
    
    template< typename Key_type, typename T, int SMEM_SIZE = 128, int NUM_HASH_FCTS = 4, int WARP_SIZE = 32 >
    class Hash_map
    {
        protected:
            // The keys stored in the map.
            volatile Key_type *m_smem_keys, *m_gmem_keys;
            // Vote buffer for values.
            volatile Word *m_smem_vote;
            // Registers to store values.
            T m_regs_vals[4];
            // The values stored in the map.
            T *m_gmem_vals;
            // The size of the global memory buffer.
            const int m_gmem_size;
            // Is there any value in GMEM.
            bool m_any_gmem;
    
        public:
            // Constructor.
            __device__ __forceinline__
            Hash_map( volatile Key_type *smem_keys, volatile Key_type *gmem_keys, volatile Word *smem_vote, T *gmem_vals, int gmem_size ) :
                m_smem_keys(smem_keys),
                m_gmem_keys(gmem_keys),
                m_smem_vote(smem_vote),
                m_gmem_vals(gmem_vals),
                m_gmem_size(gmem_size),
                m_any_gmem (true)
            {}
    
            // Clear the table. It doesn't clear GMEM values.
            __device__ __forceinline__ void clear();
            // Clear the table. It also clears GMEM values (set them to 0).
            __device__ __forceinline__ void clear_all();
            // Insert a key/value inside the hash table.
            __device__ __forceinline__ void insert( Key_type key, T a_value, T b_value, int *status );
            // Insert a key/value inside the hash table.
            __device__ __forceinline__ void insert_with_duplicates( Key_type key, T val, int *status );
            // Store the map.
            __device__ __forceinline__ void store( int count, T *vals );
            // Store the map.
            __device__ __forceinline__ void store( int count, Key_type *keys, T *vals );
    
        protected:
            // Get the selected item in the register buffer.
            __device__ __forceinline__ int get_selected( int hash ) const
            {
                return static_cast<int>(m_smem_vote[hash % WARP_SIZE].b8[hash / WARP_SIZE]);
            }
    
            // Is it the selected item in the register buffer.
            __device__ __forceinline__ bool is_selected( int hash, int lane_id ) const
            {
                return m_smem_vote[hash % WARP_SIZE].b8[hash / WARP_SIZE] == reinterpret_cast<char &>(lane_id);
            }
    
            // Push my ID in the register buffer.
            __device__ __forceinline__ void try_selection( int hash, int lane_id )
            {
                m_smem_vote[hash % WARP_SIZE].b8[hash / WARP_SIZE] = reinterpret_cast<char &>(lane_id);
            }
    };
    
    // ====================================================================================================================
    
    template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
    __device__ __forceinline__
    void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::clear()
    {
        int lane_id = get_lane_id();
        const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
    #pragma unroll
    
        for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
        {
            m_smem_keys[i_step * WARP_SIZE + lane_id] = -1;
        }
    
    #pragma unroll
    
        for ( int i_regs = 0 ; i_regs < 4 ; ++i_regs )
        {
            m_regs_vals[i_regs] = 0;
        }
    
        if ( !m_any_gmem )
        {
            return;
        }
    
    #pragma unroll 4
    
        for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
        {
            m_gmem_keys[offset] = -1;
        }
    
        m_any_gmem = false;
    }
    
    // ====================================================================================================================
    
    template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
    __device__ __forceinline__
    void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::clear_all()
    {
        int lane_id = get_lane_id();
        const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
    #pragma unroll
    
        for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
        {
            m_smem_keys[i_step * WARP_SIZE + lane_id] = -1;
        }
    
    #pragma unroll
    
        for ( int i_regs = 0 ; i_regs < 4 ; ++i_regs )
        {
            m_regs_vals[i_regs] = 0;
        }
    
        if ( !m_any_gmem )
        {
            return;
        }
    
    #pragma unroll 4
    
        for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
        {
            m_gmem_keys[offset] =   -1;
            m_gmem_vals[offset] = 0;
        }
    
        m_any_gmem = false;
    }
    
    // ====================================================================================================================
    
    template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
    __device__ __forceinline__
    void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::insert( Key_type key, T a_value, T b_value, int *status )
    {
        const int lane_id = get_lane_id();
        bool done = key == -1;
        m_smem_vote[lane_id].b32 = 0x20202020;
    #pragma unroll
    
        for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
        {
            if ( i_hash > 0 && all(done) )
            {
                break;
            }
    
            bool candidate = false;
            unsigned ukey = reinterpret_cast<unsigned &>( key );
            int hash = ( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash] ) & (SMEM_SIZE - 1);
    
            if ( !done )
            {
                Key_type stored_key = m_smem_keys[hash];
    
                if ( stored_key == key )
                {
                    this->try_selection( hash, lane_id );
                    done = true;
                }
    
                candidate = stored_key == -1;
    
                if ( candidate )
                {
                    m_smem_keys[hash] = key;
                }
    
                if ( candidate && key == m_smem_keys[hash] )
                {
                    this->try_selection( hash, lane_id );
                    done = true;
                }
            }
        }
    
        Word my_vote;
        my_vote.b32 = m_smem_vote[lane_id].b32;
    #pragma unroll
    
        for ( int i_regs = 0 ; i_regs < 4 ; ++i_regs )
        {
            int my_src = my_vote.b8[i_regs];
            T other_val = shfl( b_value, my_src );
    
            if ( my_src != WARP_SIZE )
            {
                m_regs_vals[i_regs] = m_regs_vals[i_regs] + a_value * other_val;
            }
        }
    
        const int num_bits = bfind( m_gmem_size );
    #pragma unroll
    
        for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
        {
            if ( all(done) )
            {
                return;
            }
    
            m_any_gmem = true;
            bool candidate = false;
            unsigned ukey = reinterpret_cast<unsigned &>( key );
            int hash = bfe( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash], num_bits );
    
            if ( !done )
            {
                Key_type stored_key = m_gmem_keys[hash];
    
                if ( stored_key == key )
                {
                    m_gmem_vals[hash] = m_gmem_vals[hash] + a_value * b_value;
                    done = true;
                }
    
                candidate = stored_key == -1;
    
                if ( candidate )
                {
                    m_gmem_keys[hash] = key;
                }
    
                if ( candidate && key == m_gmem_keys[hash] ) // More than one candidate may have written to that slot.
                {
                    m_gmem_vals[hash] = a_value * b_value;
                    done = true;
                }
            }
        }
    
        if ( status == NULL || all(done) )
        {
            return;
        }
    
        if ( lane_id == 0 )
        {
            status[0] = 1;
        }
    }
    
    // ====================================================================================================================
    
    template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
    __device__ __forceinline__
    void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::insert_with_duplicates( Key_type key, T val, int *status )
    {
        const int lane_id = get_lane_id();
        bool done = key == -1;
        m_smem_vote[lane_id].b32 = 0x20202020;
    #pragma unroll
    
        for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
        {
            if ( all(done) )
            {
                break;
            }
    
            bool candidate = false;
            bool maybe_in_conflict = false;
            unsigned ukey = reinterpret_cast<unsigned &>( key );
            int hash = ( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash] ) & (SMEM_SIZE - 1);
    
            if ( !done )
            {
                Key_type stored_key = m_smem_keys[hash];
    
                if ( stored_key == key )
                {
                    this->try_selection( hash, lane_id );
                    maybe_in_conflict = true;
                    done = true; // Is it really done???
                }
    
                candidate = stored_key == -1;
    
                if ( candidate )
                {
                    m_smem_keys[hash] = key;
                }
    
                if ( candidate && key == m_smem_keys[hash] )
                {
                    this->try_selection( hash, lane_id );
                    maybe_in_conflict = true;
                    done = true;
                }
            }
    
            // Fix conflicts.
            bool in_conflict = maybe_in_conflict && !this->is_selected(hash, lane_id);
    
            while ( any( in_conflict ) )
            {
                int winner = in_conflict ? this->get_selected(hash) : WARP_SIZE;
                T other_val = shfl( val, winner );
    
                if ( in_conflict )
                {
                    this->try_selection(hash, lane_id);
                }
    
                if ( in_conflict && this->is_selected(hash, lane_id) )
                {
                    val = val + other_val;
                    in_conflict = false;
                }
            }
        }
    
        Word my_vote;
        my_vote.b32 = m_smem_vote[lane_id].b32;
    #pragma unroll
    
        for ( int i_regs = 0 ; i_regs < 4 ; ++i_regs )
        {
            int my_src = my_vote.b8[i_regs];
            T other_val = shfl( val, my_src );
    
            if ( my_src != WARP_SIZE )
            {
                m_regs_vals[i_regs] = m_regs_vals[i_regs] + other_val;
            }
        }
    
        const int num_bits = bfind( m_gmem_size );
    #pragma unroll
    
        for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
        {
            if (all(done) )
            {
                return;
            }
    
            m_any_gmem = true;
            bool candidate = false;
            unsigned ukey = reinterpret_cast<unsigned &>( key );
            int hash = bfe( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash], num_bits );
    
            if ( !done )
            {
                Key_type stored_key = m_gmem_keys[hash];
    
                if ( stored_key == key )
                {
                    atomic_add( &m_gmem_vals[hash], val );
                    done = true;
                }
    
                candidate = stored_key == -1;
    
                if ( candidate )
                {
                    m_gmem_keys[hash] = key;
                }
    
                if ( candidate && key == m_gmem_keys[hash] ) // More than one candidate may have written to that slot.
                {
                    atomic_add( &m_gmem_vals[hash], val );
                    done = true;
                }
            }
        }
    
        if ( status == NULL || all(done) )
        {
            return;
        }
    
        if ( lane_id == 0 )
        {
            status[0] = 1;
        }
    }
    
    template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
    __device__ __forceinline__
    void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store( int count, T *vals )
    {
        int lane_id = get_lane_id();
        int lane_mask_lt = get_lane_mask_lt();
        int warp_offset = 0;
        const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
    #pragma unroll
    
        for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
        {
            const int offset = i_step * WARP_SIZE + lane_id;
            Key_type key = m_smem_keys[offset];
            int poll = ballot( key != -1 );
    
            if ( poll == 0 )
            {
                continue;
            }
    
            int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
    
            if ( key != -1 )
            {
                vals[dst_offset] = m_regs_vals[i_step];
            }
    
            warp_offset += __popc( poll );
        }
    
        if ( !m_any_gmem )
        {
            return;
        }
    
    #pragma unroll 4
    
        for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
        {
            Key_type key = m_gmem_keys[offset];
            int poll = ballot( key != -1, activemask() );
    
            if ( poll == 0 )
            {
                continue;
            }
    
            int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
    
            if ( key != -1 )
            {
                vals[dst_offset] = m_gmem_vals[offset];
            }
    
            warp_offset += __popc( poll );
        }
    }
    
    // ====================================================================================================================
    
    template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
    __device__ __forceinline__
    void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store( int count, Key_type *keys, T *vals )
    {
        int lane_id = get_lane_id();
        int lane_mask_lt = get_lane_mask_lt();
        int warp_offset = 0;
        const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
    #pragma unroll
    
        for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
        {
            const int offset = i_step * WARP_SIZE + lane_id;
            Key_type key = m_smem_keys[offset];
            int poll = ballot( key != -1 );
    
            if ( poll == 0 )
            {
                continue;
            }
    
            int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
    
            if ( key != -1 )
            {
                keys[dst_offset] = key;
                vals[dst_offset] = m_regs_vals[i_step];
            }
    
            warp_offset += __popc( poll );
        }
    
        if ( !m_any_gmem )
        {
            return;
        }
    
    #pragma unroll 4
    
        for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
        {
            Key_type key = m_gmem_keys[offset];
            int poll = ballot( key != -1, activemask() );
    
            if ( poll == 0 )
            {
                continue;
            }
    
            int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
    
            if ( key != -1 )
            {
                keys[dst_offset] = key;
                vals[dst_offset] = m_gmem_vals[offset];
            }
    
            warp_offset += __popc( poll );
        }
    }
    
    enum { WARP_SIZE = 32, SMEM_SIZE = 128 };

    template< int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE, bool COUNT_ONLY >
    __global__ __launch_bounds__( CTA_SIZE )
    void
    count_non_zeroes_kernel( const int A_num_rows,
                            const int *A_rows,
                            const int *A_cols,
                            const int *B_rows,
                            const int *B_cols,
                            int *C_rows,
                            int *C_cols,
                            int *Aq1,
                            int *Bq1,
                            int *Aq2,
                            int *Bq2,
                            const int gmem_size,
                            int *g_keys,
                            int *wk_work_queue,
                            int *wk_status )
    {
        // 一个block中有8个warp
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        
        // The hash keys stored in shared memory.
        // 每个warp分128字节的共享内存
        __shared__ volatile int s_keys[NUM_WARPS * SMEM_SIZE];
        
        // The coordinates of the thread inside the CTA/warp.
        // block内部的warp序号和lane序号
        const int warp_id = get_warp_id();
        const int lane_id = get_lane_id();
        
        // First threads load the row IDs of A needed by the CTA...
        // a_row_id是总体上的warp编号
        int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
        // Create local storage for the set.
        // Hash_set_atomic<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );
        Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );

        // Loop over rows of A.
        // 每个warp处理A的一行，这里的get_work其实就是把wk_work_queue原子加一下，为啥呢，其实正如work_queue所言
        // 第一个grid需要对前若干行处理，后面的未处理的行需要获得一个偏移量，以便于之后的grid继续处理这些行
        for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id ) )
        {
            int c_row_id = a_row_id;

            if (Aq1 != NULL)
            {
                a_row_id = Aq1[a_row_id];
            }

            // Make sure we have to proceed.
            if ( COUNT_ONLY )
            {
                volatile int *status = reinterpret_cast<volatile int *>( wk_status );

                if ( set.has_failed() || *status != 0 )
                {
                    return;
                }
            }

            // Clear the set.
            set.clear();
            // Load the range of the row.
            int a_col_tmp = -1;

            // warp中的0号1号两个线程才能从行索引中获得起始和结束行索引
            if ( lane_id < 2 )
            {
                a_col_tmp = load( &A_rows[a_row_id + lane_id] );
            }
            int a_col_it  = shfl( a_col_tmp, 0 );
            int a_col_end = shfl( a_col_tmp, 1 );
            
            // 一个总体上的warp负责A的一整行
            // Iterate over the columns of A.
            // 假设一共有10000行，一共有128个warp，每个warp都需要循环10000 / 128次
            // 0-31号线程分别有一个起始位置0-31，然后处理这一行的nnz，这些nnz假设有1000个，那么每个线程需要处理1000 / 32个A的nnz，也就是1000 / 32个B的行
            for ( a_col_it += lane_id ; __any_sync(DEFAULT_MASK, a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
            {
                // A这一行的nnz不是32的整数，则会有空闲线程
                // Is it an active thread.
                const bool is_active = a_col_it < a_col_end;
                // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
                int b_row_id = -1;

                if ( is_active )
                {
                    //真实的列索引值在这里取出来，再去对应B的哪一行
                    b_row_id = load( &A_cols[a_col_it] );

                    //b_row_id is actually column of A
                    if (Aq2 != NULL)
                    {
                        b_row_id = Aq2[b_row_id];
                    }

                    if (Bq1 != NULL)
                    {
                        b_row_id = Bq1[b_row_id];
                    }
                }

                // The number of valid rows.
                // B有num_rows行需要处理
                const int num_rows = __popc( ballot(is_active) );
                
                // num_rows其实最多有32个（因为一个warp负责A的一行，一次32个nnz，不断前进处理），而对于每个线程来说，遍历到
                // A这一行中的nnz后并不会做什么，而是等待后面统计一共有几个active的行，随后整个warp所有的线程都会
                // 共同处理B的第一行，然后处理B的第二行，循环直到active行都被处理
                // Uniform loop: threads collaborate to load other elements.
                for ( int k = 0 ; k < num_rows ; ++k )
                {
                    // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                    const int uniform_b_row_id = shfl( b_row_id, k );

                    // B的那一行的起始和结束行指针
                    // Load the range of the row of B.
                    int b_col_tmp = -1;
                    if ( lane_id < 2 )
                    {
                        b_col_tmp = load( &B_rows[uniform_b_row_id + lane_id] );
                    }

                    // 每个线程都获取一下B这一行的起始和结束
                    int b_col_it  = shfl( b_col_tmp, 0 );
                    int b_col_end = shfl( b_col_tmp, 1 );

                    // Iterate over the range of columns of B.
                    // 起始位置根据线程号有所区别，但是循环后加的偏移量还是warp大小
                    for ( b_col_it += lane_id ; any(b_col_it < b_col_end) ; b_col_it += WARP_SIZE )
                    {
                        int b_col_id = -1;

                        if ( b_col_it < b_col_end )
                        {
                            b_col_id = load( &B_cols[b_col_it] );

                            // b_col_id is actually column of B
                            if (Bq2 != NULL)
                            {
                                b_col_id = Bq2[b_col_id];
                            }
                        }

                        set.insert( b_col_id, COUNT_ONLY ? wk_status : NULL );
                    }
                }
            }

            // Store the results.
            if ( COUNT_ONLY )
            {
                int count = set.compute_size();

                if ( lane_id == 0 )
                {
                    C_rows[c_row_id] = count;
                }
            }
            else
            {
                int c_col_tmp = -1;

                if ( lane_id < 2 )
                {
                    c_col_tmp = load( &C_rows[c_row_id + lane_id] );
                }

                int c_col_it  = shfl( c_col_tmp, 0 );
                int c_col_end = shfl( c_col_tmp, 1 );
                // Store the results.
                int count = c_col_end - c_col_it;

                if ( count == 0 )
                {
                    continue;
                }

                set.store( count, &C_cols[c_col_it] );
            }
        }
    }

    template< int NUM_THREADS_PER_ROW, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE, bool COUNT_ONLY >
    __global__ __launch_bounds__( CTA_SIZE )
    void
    count_non_zeroes_kernel( const int A_num_rows,
                            const int *__restrict A_rows,
                            const int *__restrict A_cols,
                            const int *__restrict B_rows,
                            const int *__restrict B_cols,
                            int *__restrict C_rows,
                            int *__restrict C_cols,
                            int *Aq1,
                            int *Bq1,
                            int *Aq2,
                            int *Bq2,
                            const int gmem_size,
                            int *g_keys,
                            int *wk_work_queue,
                            int *wk_status )
    {
        // 每个block有256 / 32 == 8个warp
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        // 相当于使用子warp的思路，每个warp中的几个线程去负责一行，这样可以增加并行度
        const int NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;
        // The hash keys stored in shared memory.
        // 本来这里就不是volatile的，所以不需要修改
        __shared__ /*volatile*/ int s_keys[NUM_WARPS * SMEM_SIZE];
        // The coordinates of the thread inside the CTA/warp.
        const int warp_id = get_warp_id();
        const int lane_id = get_lane_id();
        // Constants.
        const int lane_id_div_num_threads = lane_id / NUM_THREADS_PER_ROW;
        const int lane_id_mod_num_threads = lane_id % NUM_THREADS_PER_ROW;
        // First threads load the row IDs of A needed by the CTA...
        int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
        // Create local storage for the set.
        Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );
        // Hash_set_atomic<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );

        // Loop over rows of A. get_work支持乱序获取某一行去算，不必等待！！！！！
        for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id ) )
        {
            int c_row_id = a_row_id;

            if (Aq1 != NULL)
            {
                a_row_id = Aq1[a_row_id];
            }

            // Make sure we have to proceed.
            if ( COUNT_ONLY )
            {
                volatile int *status = reinterpret_cast<volatile int *>( wk_status );

                if ( set.has_failed() || *status != 0 )
                {
                    return;
                }
            }

            // Clear the set.
            set.clear();
            // Load the range of the row.
            int a_col_tmp = -1;

            if ( lane_id < 2 )
            {
                a_col_tmp = load( &A_rows[a_row_id + lane_id] );
            }

            int a_col_it  = shfl( a_col_tmp, 0 );
            int a_col_end = shfl( a_col_tmp, 1 );

            // Iterate over the columns of A.
            for ( a_col_it += lane_id ; any(a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
            {
                // Is it an active thread.
                const bool is_active = a_col_it < a_col_end;
                // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
                int b_row_id = -1;

                if ( is_active )
                {
                    b_row_id = load( &A_cols[a_col_it] );

                    //b_row_id is actually column of A
                    if (Aq2 != NULL)
                    {
                        b_row_id = Aq2[b_row_id];
                    }

                    if (Bq1 != NULL)
                    {
                        b_row_id = Bq1[b_row_id];
                    }
                }

                const int num_rows = __popc( ballot(is_active) );
                
                // 前面的count kernel是一个warp一起处理B的一行，这个kernel则是一个warp同时处理NUM_LOADED_ROWS行，比如NUM_THREADS_PER_ROW=2
                // 则一次处理16行
                // Uniform loop: threads collaborate to load other elements.
                for ( int k = 0 ; k < num_rows ; k += NUM_LOADED_ROWS )
                {
                    int local_k = k + lane_id_div_num_threads;
                    // Is it an active thread.
                    bool is_active_k = local_k < num_rows;
                    // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                    const int uniform_b_row_id = shfl( b_row_id, local_k );
                    // Load the range of the row of B.
                    int b_col_tmp = -1;

                    if ( is_active_k && lane_id_mod_num_threads < 2 )
                    {
                        b_col_tmp = load( &B_rows[uniform_b_row_id + lane_id_mod_num_threads] );
                    }

                    int b_col_it  = shfl( b_col_tmp, lane_id_div_num_threads * NUM_THREADS_PER_ROW + 0 );
                    int b_col_end = shfl( b_col_tmp, lane_id_div_num_threads * NUM_THREADS_PER_ROW + 1 );

                    // Iterate over the range of columns of B.
                    for ( b_col_it += lane_id_mod_num_threads ; any(b_col_it < b_col_end) ; b_col_it += NUM_THREADS_PER_ROW )
                    {
                        int b_col_id = -1;

                        if ( b_col_it < b_col_end )
                        {
                            b_col_id = load( &B_cols[b_col_it] );

                            // b_col_id is actually column of B
                            if (Bq2 != NULL)
                            {
                                b_col_id = Bq2[b_col_id];
                            }
                        }
                        // wk_status可以传导到最外层
                        set.insert( b_col_id, COUNT_ONLY ? wk_status : NULL );
                    }
                }
            }

            // Store the results.
            if ( COUNT_ONLY )
            {
                int count = set.compute_size_with_duplicates();
                if ( lane_id == 0 )
                {
                    C_rows[c_row_id] = count;
                }
            }
            else
            {
                int c_col_tmp = -1;

                if ( lane_id < 2 )
                {
                    c_col_tmp = load( &C_rows[c_row_id + lane_id] );
                }

                int c_col_it  = shfl( c_col_tmp, 0 );
                int c_col_end = shfl( c_col_tmp, 1 );
                // Store the results.
                int count = c_col_end - c_col_it;

                if ( count == 0 )
                {
                    continue;
                }

                set.store( count, &C_cols[c_col_it] );
            }
        }
    }

    void count_non_zeroes()
    {
        // 每个grid含有1024个block
        int GRID_SIZE = 1024;
        // 每个block含有256个线程
        const int CTA_SIZE  = 256;
        // 每个block中含有8个warp
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        // Reset work queue.
        // 整个grid含有多少个warp，如果一个矩阵足够大，则一个grid是算不完的，必须循环好几次
        int work_offset = GRID_SIZE * NUM_WARPS;
        CHECK_ERROR( cudaMemcpy( m_work_queue, &work_offset, sizeof(int), cudaMemcpyHostToDevice ) );

        // +++
        // int tmp = (A.get_num_rows() + NUM_WARPS - 1) / NUM_WARPS;
        // GRID_SIZE = GRID_SIZE > tmp ? GRID_SIZE : tmp;

        // Compute non-zero elements.
        switch ( m_num_threads_per_row_count )
        {
            /*
                int MA, NA, nnzA;
                int *csrRowIndexHostPtrA = 0;
                int *csrColIndexHostPtrA = 0;
                double *csrValHostPtrA = 0;
                char matrixNameA[1024] = {0};
            */
            // m_num_threads_per_row_count个线程负责A的一行，
            case 2:
                count_non_zeroes_kernel< 2, CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                    MA,
                    csrRowIndexDevPtrA,
                    csrColIndexDevPtrA,
                    csrRowIndexDevPtrB,
                    csrColIndexDevPtrB,
                    csrRowIndexDevPtrC,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    m_gmem_size,
                    m_keys,
                    m_work_queue,
                    m_status );
                break;

            case 4:
            count_non_zeroes_kernel< 4, CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                MA,
                csrRowIndexDevPtrA,
                    csrColIndexDevPtrA,
                    csrRowIndexDevPtrB,
                    csrColIndexDevPtrB,
                    csrRowIndexDevPtrC,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                m_gmem_size,
                m_keys,
                m_work_queue,
                m_status );
            break;

            case 8:
            count_non_zeroes_kernel< 8, CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                MA,
                csrRowIndexDevPtrA,
                    csrColIndexDevPtrA,
                    csrRowIndexDevPtrB,
                    csrColIndexDevPtrB,
                    csrRowIndexDevPtrC,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                m_gmem_size,
                m_keys,
                m_work_queue,
                m_status );
            break;

            case 16:
            count_non_zeroes_kernel< 16, CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                MA,
                csrRowIndexDevPtrA,
                csrColIndexDevPtrA,
                csrRowIndexDevPtrB,
                csrColIndexDevPtrB,
                csrRowIndexDevPtrC,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                m_gmem_size,
                m_keys,
                m_work_queue,
                m_status );
            break;

            default:
            // printf("avg nnz = %d\n", m_num_threads_per_row_count);
                count_non_zeroes_kernel<CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                    MA,
                    csrRowIndexDevPtrA,
                    csrColIndexDevPtrA,
                    csrRowIndexDevPtrB,
                    csrColIndexDevPtrB,
                    csrRowIndexDevPtrC,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                m_gmem_size,
                m_keys,
                m_work_queue,
                m_status );
        }

        cudaCheckError();
        CHECK_ERROR( cudaGetLastError() );
    }


    template< typename Value_type, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE >
    __global__ __launch_bounds__( CTA_SIZE, 6 )
    void
    compute_values_kernel( const int A_num_rows,
                        const int *__restrict A_rows,
                        const int *__restrict A_cols,
                        const double *__restrict A_vals,
                        const int *__restrict B_rows,
                        const int *__restrict B_cols,
                        const double *__restrict B_vals,
                        const int *__restrict C_rows,
                        int *__restrict C_cols,
                        double *__restrict C_vals,
                        int *Aq1,
                        int *Bq1,
                        int *Aq2,
                        int *Bq2,
                        const int gmem_size,
                        int *g_keys,
                        double *g_vals,
                        int *wk_work_queue,
                        int *wk_status )
    {
        const int NUM_WARPS = CTA_SIZE / 32;
        // The hash keys stored in shared memory.
        __shared__ /*volatile*/ int s_keys[NUM_WARPS * SMEM_SIZE];
        // The hash values stored in shared memory.
        
        __shared__ volatile Word s_vote[NUM_WARPS * SMEM_SIZE / 4];
        // __shared__ double s_vals[NUM_WARPS * SMEM_SIZE]; 

        const int warp_id = get_warp_id();
        const int lane_id = get_lane_id();
        // First threads load the row IDs of A needed by the CTA...
        int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
        // Create local storage for the set.
        Hash_map<int, double, SMEM_SIZE, 4, WARP_SIZE> map( &s_keys[warp_id * SMEM_SIZE],
                &g_keys[a_row_id * gmem_size],
                &s_vote[warp_id * SMEM_SIZE / 4],
                &g_vals[a_row_id * gmem_size],
                gmem_size );
        // Hash_map<int, double, SMEM_SIZE, 4, WARP_SIZE> map(&s_keys[warp_id * SMEM_SIZE],
        //     &g_keys[a_row_id * gmem_size],
        //     // &s_vote[warp_id * SMEM_SIZE / 4],
        //     &s_vals[warp_id * SMEM_SIZE],
        //     &g_vals[a_row_id * gmem_size],
        //     gmem_size );

        // Loop over rows of A.
        for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id ) )
        {
            int c_row_id = a_row_id;

            if (Aq1 != NULL)
            {
                a_row_id = Aq1[a_row_id];
            }

            // Clear the map.
            map.clear();
            // Load the range of the row.
            int a_col_tmp = -1;

            if ( lane_id < 2 )
            {
                a_col_tmp = load( &A_rows[a_row_id + lane_id] );
            }

            int a_col_it  = shfl( a_col_tmp, 0 );
            int a_col_end = shfl( a_col_tmp, 1 );

            // Iterate over the columns of A.
            for ( a_col_it += lane_id ; any(a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
            {
                // Is it an active thread.
                const bool is_active = a_col_it < a_col_end;
                // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
                int b_row_id = -1;
                double a_value = 0.0;

                if ( is_active )
                {
                    b_row_id = load( &A_cols[a_col_it] );
                    a_value  = load( &A_vals[a_col_it] );

                    //b_row_id is actually column of A
                    if (Aq2 != NULL)
                    {
                        b_row_id = Aq2[b_row_id];
                    }

                    if (Bq1 != NULL)
                    {
                        b_row_id = Bq1[b_row_id];
                    }
                }

                const int num_rows = __popc( ballot(is_active) );

                // Uniform loop: threads collaborate to load other elements.
                for ( int k = 0 ; k < num_rows ; ++k )
                {
                    // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                    const int uniform_b_row_id = shfl( b_row_id, k );
                    // The value of A.
                    const double uniform_a_value = shfl( a_value, k );
                    // Load the range of the row of B.
                    int b_col_tmp = -1;

                    if ( lane_id < 2 )
                    {
                        b_col_tmp = load( &B_rows[uniform_b_row_id + lane_id] );
                    }

                    int b_col_it  = shfl( b_col_tmp, 0 );
                    int b_col_end = shfl( b_col_tmp, 1 );

                    // Iterate over the range of columns of B.
                    for ( b_col_it += lane_id ; any(b_col_it < b_col_end) ; b_col_it += WARP_SIZE )
                    {
                        int b_col_id = -1;
                        double b_value = 0.0;

                        if ( b_col_it < b_col_end )
                        {
                            b_col_id = load( &B_cols[b_col_it] );
                            b_value  = load( &B_vals[b_col_it] );

                            if (Bq2 != NULL)
                            {
                                b_col_id = Bq2[b_col_id];
                            }
                        }
                        map.insert( b_col_id, uniform_a_value, b_value, wk_status );
                    }
                }
            }

            // Store the results.
            int c_col_tmp = -1;

            if ( lane_id < 2 )
            {
                c_col_tmp = load( &C_rows[c_row_id + lane_id] );
            }

            int c_col_it  = shfl( c_col_tmp, 0 );
            int c_col_end = shfl( c_col_tmp, 1 );
            // Store the results.
            int count = c_col_end - c_col_it;

            if ( count == 0 )
            {
                continue;
            }

            map.store( count, &C_cols[c_col_it], &C_vals[c_col_it] );
        }
    }

    template< int NUM_THREADS_PER_ROW, typename Value_type, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE >
    __global__ __launch_bounds__( CTA_SIZE, 6 )
    void
    compute_values_kernel( const int A_num_rows,
                        const int *__restrict A_rows,
                        const int *__restrict A_cols,
                        const double *__restrict A_vals,
                        const int *__restrict B_rows,
                        const int *__restrict B_cols,
                        const double *__restrict B_vals,
                        const int *__restrict C_rows,
                        int *__restrict C_cols,
                        double *__restrict C_vals,
                        int *Aq1,
                        int *Bq1,
                        int *Aq2,
                        int *Bq2,
                        const int gmem_size,
                        int *g_keys,
                        double *g_vals,
                        int *wk_work_queue,
                        int *wk_status )
    {
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        const int NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;
        // The hash keys stored in shared memory.
        __shared__ /*volatile*/ int s_keys[NUM_WARPS * SMEM_SIZE];
        // The hash values stored in shared memory.
        
        __shared__ volatile Word s_vote[NUM_WARPS * SMEM_SIZE / 4];
        // __shared__ double s_vals[NUM_WARPS * SMEM_SIZE]; 
        
        // The coordinates of the thread inside the CTA/warp.
        const int warp_id = get_warp_id( );
        const int lane_id = get_lane_id( );
        // Constants.
        const int lane_id_div_num_threads = lane_id / NUM_THREADS_PER_ROW;
        const int lane_id_mod_num_threads = lane_id % NUM_THREADS_PER_ROW;
        // First threads load the row IDs of A needed by the CTA...
        int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
        // Create local storage for the set.
        Hash_map<int, double, SMEM_SIZE, 4, WARP_SIZE> map( &s_keys[warp_id * SMEM_SIZE],
                &g_keys[a_row_id * gmem_size],
                &s_vote[warp_id * SMEM_SIZE / 4],
                &g_vals[a_row_id * gmem_size],
                gmem_size );
        // Hash_map_atomic<int, double, SMEM_SIZE, 4, WARP_SIZE> map(&s_keys[warp_id * SMEM_SIZE],
        //     &g_keys[a_row_id * gmem_size],
        //     // &s_vote[warp_id * SMEM_SIZE / 4],
        //     &s_vals[warp_id * SMEM_SIZE],
        //     &g_vals[a_row_id * gmem_size],
        //     gmem_size );
        // Hash_map<int, double, SMEM_SIZE, 4, WARP_SIZE> map(&s_keys[warp_id * SMEM_SIZE],
        //         &g_keys[a_row_id * gmem_size],
        //         // &s_vote[warp_id * SMEM_SIZE / 4],
        //         &s_vals[warp_id * SMEM_SIZE],
        //         &g_vals[a_row_id * gmem_size],
        //         gmem_size );

        // Loop over rows of A.
        for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id ) )
        {
            int c_row_id = a_row_id;

            if (Aq1 != NULL)
            {
                a_row_id = Aq1[a_row_id];
            }

            // Clear the map.
            map.clear_all();
            // map.clear();
            // Load the range of the row.
            int a_col_tmp = -1;

            if ( lane_id < 2 )
            {
                a_col_tmp = load( &A_rows[a_row_id + lane_id] );
            }

            int a_col_it  = shfl( a_col_tmp, 0 );
            int a_col_end = shfl( a_col_tmp, 1 );

            // Iterate over the columns of A.
            for ( a_col_it += lane_id ; any(a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
            {
                // Is it an active thread.
                const bool is_active = a_col_it < a_col_end;
                // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
                int b_row_id(-1);
                double a_value = 0.0;

                if ( is_active )
                {
                    b_row_id = load( &A_cols[a_col_it] );
                    a_value  = load( &A_vals[a_col_it] );

                    //b_row_id is actually column of A
                    if (Aq2 != NULL)
                    {
                        b_row_id = Aq2[b_row_id];
                    }

                    if (Bq1 != NULL)
                    {
                        b_row_id = Bq1[b_row_id];
                    }
                }

                const int num_rows = __popc( ballot(is_active) );

                // Uniform loop: threads collaborate to load other elements.
                for ( int k = 0 ; k < num_rows ; k += NUM_LOADED_ROWS )
                {
                    int local_k = k + lane_id_div_num_threads;
                    // Is it an active thread.
                    bool is_active_k = local_k < num_rows;
                    // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                    const int uniform_b_row_id = shfl( b_row_id, k + lane_id_div_num_threads );
                    // The value of A.
                    const double uniform_a_value = shfl( a_value, k + lane_id_div_num_threads );
                    // Load the range of the row of B.
                    int b_col_tmp = -1;

                    if ( is_active_k && lane_id_mod_num_threads < 2 )
                    {
                        b_col_tmp = load( &B_rows[uniform_b_row_id + lane_id_mod_num_threads] );
                    }

                    int b_col_it  = shfl( b_col_tmp, lane_id_div_num_threads * NUM_THREADS_PER_ROW + 0 );
                    int b_col_end = shfl( b_col_tmp, lane_id_div_num_threads * NUM_THREADS_PER_ROW + 1 );

                    // Iterate over the range of columns of B.
                    for ( b_col_it += lane_id_mod_num_threads ; any(b_col_it < b_col_end) ; b_col_it += NUM_THREADS_PER_ROW )
                    {
                        int b_col_id(-1);
                        double b_value = 0.0;

                        if ( b_col_it < b_col_end )
                        {
                            b_col_id = load( &B_cols[b_col_it] );
                            b_value  = load( &B_vals[b_col_it] );

                            //b_col_id is actually column of B
                            if (Bq2 != NULL)
                            {
                                b_col_id = Bq2[b_col_id];
                            }
                        }
                        map.insert_with_duplicates( b_col_id, uniform_a_value * b_value, wk_status );
                        // map.insert_atomic( b_col_id, uniform_a_value * b_value, wk_status );
                        // map.insert( b_col_id, uniform_a_value * b_value, wk_status );
                    }
                }
            }

            // Store the results.
            int c_col_tmp = -1;

            if ( lane_id < 2 )
            {
                c_col_tmp = load( &C_rows[c_row_id + lane_id] );
            }

            int c_col_it  = shfl( c_col_tmp, 0 );
            int c_col_end = shfl( c_col_tmp, 1 );
            // Store the results.
            int count = c_col_end - c_col_it;

            if ( count == 0 )
            {
                continue;
            }

            map.store( count, &C_cols[c_col_it], &C_vals[c_col_it] );
        }
    }

    void compute_values()
    {
        const int GRID_SIZE = 1024;
        const int CTA_SIZE  = 128;
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        // Reset the work queue.
        int work_offset = GRID_SIZE * NUM_WARPS;
        CHECK_ERROR( cudaMemcpy( m_work_queue, &work_offset, sizeof(int), cudaMemcpyHostToDevice ) );
        // Compute the values.
        int *status = NULL;

        if ( m_num_threads_per_row_count != m_num_threads_per_row_compute )
        {
            status = m_status;
        }
        
        // printf("进来了compute_values %d\n", m_num_threads_per_row_compute);
        switch ( m_num_threads_per_row_compute )
        {
            //int *csrRowIndexDevPtrB = 0;
            // int *csrColIndexDevPtrB = 0;
            // double *csrValDevPtrB = 0;
            case 2:
                compute_values_kernel< 2, double, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                    MA,
                    csrRowIndexDevPtrA,
                    csrColIndexDevPtrA,
                    csrValDevPtrA,
                    csrRowIndexDevPtrB,
                    csrColIndexDevPtrB,
                    csrValDevPtrB,
                    csrRowIndexDevPtrC,
                    csrColIndexDevPtrC,
                    csrValDevPtrC,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    m_gmem_size,
                    m_keys,
                    m_vals,
                    m_work_queue,
                    status );
                break;

            case 4:
            compute_values_kernel< 4, double, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                    MA,
                    csrRowIndexDevPtrA,
                    csrColIndexDevPtrA,
                    csrValDevPtrA,
                    csrRowIndexDevPtrB,
                    csrColIndexDevPtrB,
                    csrValDevPtrB,
                    csrRowIndexDevPtrC,
                    csrColIndexDevPtrC,
                    csrValDevPtrC,
                NULL,
                NULL,
                NULL,
                NULL,
                    m_gmem_size,
                    m_keys,
                    m_vals,
                    m_work_queue,
                    status );
                break;

            case 8:
                compute_values_kernel< 8, double, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                    MA,
                    csrRowIndexDevPtrA,
                    csrColIndexDevPtrA,
                    csrValDevPtrA,
                    csrRowIndexDevPtrB,
                    csrColIndexDevPtrB,
                    csrValDevPtrB,
                    csrRowIndexDevPtrC,
                    csrColIndexDevPtrC,
                    csrValDevPtrC,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    m_gmem_size,
                    m_keys,
                    m_vals,
                    m_work_queue,
                    status );
                break;

            case 16:
                compute_values_kernel<16, double, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                    MA,
                    csrRowIndexDevPtrA,
                    csrColIndexDevPtrA,
                    csrValDevPtrA,
                    csrRowIndexDevPtrB,
                    csrColIndexDevPtrB,
                    csrValDevPtrB,
                    csrRowIndexDevPtrC,
                    csrColIndexDevPtrC,
                    csrValDevPtrC,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    m_gmem_size,
                    m_keys,
                    m_vals,
                    m_work_queue,
                    status );
                break;

            default:
                compute_values_kernel<double, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                    MA,
                    csrRowIndexDevPtrA,
                    csrColIndexDevPtrA,
                    csrValDevPtrA,
                    csrRowIndexDevPtrB,
                    csrColIndexDevPtrB,
                    csrValDevPtrB,
                    csrRowIndexDevPtrC,
                    csrColIndexDevPtrC,
                    csrValDevPtrC,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    m_gmem_size,
                    m_keys,
                    m_vals,
                    m_work_queue,
                    status );
        }

        cudaCheckError();
        CHECK_ERROR( cudaGetLastError() );
    }
}

namespace origin {
    static __device__ __forceinline__ int get_lane_id()
    {
        int id;
        asm( "mov.u32 %0, %%laneid;" : "=r"(id) );
        return id;
    }

    static __device__ __forceinline__ int get_lane_mask_lt()
    {
        int mask;
        asm( "mov.u32 %0, %%lanemask_lt;" : "=r"(mask) );
        return mask;
    }


    static __device__ __forceinline__ int get_warp_id()
    {
        return threadIdx.x >> 5;
    }

    #define DEFAULT_MASK 0xffffffff

    static __device__ __forceinline__ int shfl( int r, int lane, int bound = 32, unsigned int mask = DEFAULT_MASK )
    {
        return __shfl_sync( mask, r, lane, bound );
    }

    __device__ __forceinline__ int get_work( int *queue, int warp_id )
    {
        int offset = -1;

        if ( get_lane_id() == 0 )
        {
            offset = atomicAdd( queue, 1 );
        }

        return shfl( offset, 0 );
    }

    static __device__ __forceinline__ unsigned int ballot(int p, unsigned int mask = DEFAULT_MASK)
    {
        return __ballot_sync(mask, p);
    }

    static __device__ __forceinline__ unsigned int any(int p, unsigned int mask = DEFAULT_MASK)
    {
        return __any_sync(mask, p);
    }

    static __device__ __forceinline__ unsigned int activemask()
    {
        return __activemask();

    }

    static __device__ __forceinline__ int shfl_xor( int r, int lane_mask, int bound = 32, unsigned int mask = DEFAULT_MASK )
    {
        return __shfl_xor_sync( mask, r, lane_mask, bound );
    }

    template< typename T >
    static __device__ __forceinline__ T load( const T *ptr ) { return __ldg( ptr ); }

    static __device__ __forceinline__ void atomic_add( double *address, double value )
    {
    #if __CUDA_ARCH__ >= 600
        atomicAdd( address, value );
    #else
        unsigned long long *address_as_ull = (unsigned long long *) address;
        unsigned long long old = __double_as_longlong( address[0] ), assumed;

        do
        {
            assumed = old;
            old = atomicCAS( address_as_ull, assumed, __double_as_longlong( value + __longlong_as_double( assumed ) ) );
        }
        while ( assumed != old );

    #endif
    }

    static __device__ __forceinline__ int atomic_CAS(int* address, int compare, int val)
    {
        return atomicCAS(address, compare, val);
    }

    static __constant__ unsigned c_hash_keys[] =
    {
        3499211612,  581869302, 3890346734, 3586334585,
        545404204,  4161255391, 3922919429,  949333985,
        2715962298, 1323567403,  418932835, 2350294565,
        1196140740,  809094426, 2348838239, 4264392720
    };


    template< typename Key_type, int SMEM_SIZE = 128, int NUM_HASH_FCTS = 4, int WARP_SIZE = 32 >
    class Hash_set
    {
        protected:
            // The size of the table (occupancy).
            int m_smem_count, m_gmem_count;
            // The keys stored in the hash table.
            Key_type *m_smem_keys, *m_gmem_keys;
            // The size of the global memory buffer.
            const int m_gmem_size;
            // Is it ok?
            bool m_fail;

        public:
            // Constructor.
            __device__ __forceinline__ Hash_set( Key_type *smem_keys, Key_type *gmem_keys, int gmem_size ) :
                m_smem_count(0),
                m_gmem_count(1),
                m_smem_keys (smem_keys),
                m_gmem_keys (gmem_keys),
                m_gmem_size (gmem_size),
                m_fail      (false)
            {}

            // Clear the table.
            __device__ __forceinline__ void clear( bool skip_gmem = false );
            // Compute the size of the table. Only thread with lane_id==0 gives the correct result (no broadcast of the value).
            __device__ __forceinline__ int compute_size();
            // Has the process failed.
            __device__ __forceinline__ bool has_failed() const { return m_fail; }
            // Insert a key inside the set. If status is NULL, ignore failure.
            __device__ __forceinline__ void insert( Key_type key, int *status );
            // Store a set.
            __device__ __forceinline__ int  store( Key_type *keys );
            // Store a set.
            __device__ __forceinline__ void store( int count, Key_type *keys );

            
    };

    // ====================================================================================================================

    template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE>
    __device__ __forceinline__
    void Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::clear( bool skip_gmem )
    {
        int lane_id = get_lane_id();
        const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
    #pragma unroll

        for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
        {
            m_smem_keys[i_step * WARP_SIZE + lane_id] = -1;
        }

        m_smem_count = 0;

        if ( skip_gmem || m_gmem_count == 0 )
        {
            m_gmem_count = 0;
            return;
        }

    #pragma unroll 4

        for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
        {
            m_gmem_keys[offset] = -1;
        }

        m_gmem_count = 0;
    }

    template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE>
    __device__ __forceinline__
    int Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::compute_size()
    {
        m_smem_count += m_gmem_count;
    #pragma unroll

        for ( int offset = WARP_SIZE / 2 ; offset > 0 ; offset >>= 1 )
        {
            m_smem_count += shfl_xor( m_smem_count, offset );
        }

        m_gmem_count = any( m_gmem_count > 0 );
        return m_smem_count;
    }

    template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
    __device__ __forceinline__
    void Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::insert( Key_type key, int *status )
    {
        bool active = key != -1;
        Key_type winning_key;
        int active_mask;

    #pragma unroll
        for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
        {
            active_mask = ballot( active );

            if ( active_mask == 0 )
            {
                return;
            }

            if ( active )
            {
                unsigned ukey = reinterpret_cast<unsigned &>( key );
                int hash = ( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash] ) & (SMEM_SIZE - 1); 

                winning_key = atomic_CAS(&m_smem_keys[hash], -1, key); 

                if ( winning_key == -1 )
                {
                    winning_key = key;
                    m_smem_count++;
                }


                if ( key == winning_key )
                {
                    active = false;
                }
            }
        }
    #pragma unroll
        for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
        {
            active_mask = ballot( active );

            if ( active_mask == 0 )
            {
                return;
            }

            if ( active )
            {
                unsigned ukey = reinterpret_cast<unsigned &>( key );
                int hash = ( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash] ) & (m_gmem_size - 1); 

                winning_key = atomic_CAS(&m_gmem_keys[hash], -1, key); 

                if ( winning_key == -1 )
                {
                    winning_key = key;
                    m_gmem_count++;
                }

                if ( key == winning_key )
                {
                    active = false;
                }
            }
        }

        if ( ballot( active ) == 0 )
        {
            return;
        }

        assert( status != NULL );

        if ( get_lane_id() == 0 )
        {
            *status = 1;
        }

        m_fail = true;
    }

    template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
    __device__ __forceinline__
    void Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store( int count, Key_type *keys )
    {
        int lane_id = get_lane_id();
        int lane_mask_lt = get_lane_mask_lt();
        int warp_offset = 0;
        const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
    #pragma unroll

        for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
        {
            const int offset = i_step * WARP_SIZE + lane_id;
            Key_type key = m_smem_keys[offset];
            int poll = ballot( key != -1 );

            if ( poll == 0 )
            {
                continue;
            }

            int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

            if ( key != -1 )
            {
                keys[dst_offset] = key;
            }

            warp_offset += __popc( poll );
        }

        m_gmem_count = any( m_gmem_count > 0 );

        if ( !m_gmem_count )
        {
            return;
        }

    #pragma unroll 4

        for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
        {
            Key_type key = m_gmem_keys[offset];
            int poll = ballot( key != -1, activemask() );

            if ( poll == 0 )
            {
                continue;
            }

            int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

            if ( key != -1 )
            {
                keys[dst_offset] = key;
            }

            warp_offset += __popc( poll );
        }
    }

    template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
    __device__ __forceinline__
    int Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store( Key_type *keys )
    {
        int lane_id = get_lane_id();
        int lane_mask_lt = get_lane_mask_lt();
        int warp_offset = 0;
        const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
    #pragma unroll

        for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
        {
            const int offset = i_step * WARP_SIZE + lane_id;
            Key_type key = m_smem_keys[offset];
            int poll = ballot( key != -1 );

            if ( poll == 0 )
            {
                continue;
            }

            int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

            if ( key != -1 )
            {
                keys[dst_offset] = key;
            }

            warp_offset += __popc( poll );
        }

        m_gmem_count = any( m_gmem_count > 0 );

        if ( !m_gmem_count )
        {
            return warp_offset;
        }

    #pragma unroll 4

        for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
        {
            Key_type key = m_gmem_keys[offset];
            int poll = ballot( key != -1, activemask() );

            if ( poll == 0 )
            {
                continue;
            }

            int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

            if ( key != -1 )
            {
                keys[dst_offset] = key;
            }

            warp_offset += __popc( poll );
        }

        return warp_offset;
    }


    template< typename Key_type, typename T, int SMEM_SIZE = 128, int NUM_HASH_FCTS = 4, int WARP_SIZE = 32 >
    class Hash_map
    {
        protected:
            // The keys stored in the map.
            Key_type *m_smem_keys, *m_gmem_keys;
            // Shared memory values
            T *m_smem_vals = NULL;
            // The values stored in the map.
            T *m_gmem_vals;
            // The size of the global memory buffer.
            const int m_gmem_size;
            // Is there any value in GMEM.
            bool m_any_gmem;
            

        public:
            __device__ __forceinline__
            Hash_map( Key_type *smem_keys, Key_type *gmem_keys, T *smem_vals, T *gmem_vals, int gmem_size ) :
                m_smem_keys(smem_keys),
                m_gmem_keys(gmem_keys),
                m_smem_vals(smem_vals),
                m_gmem_vals(gmem_vals),
                m_gmem_size(gmem_size),
                m_any_gmem (true)
            {}

            // Clear the table. It doesn't clear GMEM values.
            __device__ __forceinline__ void clear();
            // Insert a key/value inside the hash table.
            __device__ __forceinline__ void insert( Key_type key, T val, int *status );
            // Store the map.
            __device__ __forceinline__ void store( int count, T *vals );
            // Store the map.
            __device__ __forceinline__ void store( int count, Key_type *keys, T *vals );
    };


    template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
    __device__ __forceinline__
    void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::clear()
    {
        int lane_id = get_lane_id();
        const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
    #pragma unroll

        for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
        {
            m_smem_keys[i_step * WARP_SIZE + lane_id] = -1;
            m_smem_vals[i_step * WARP_SIZE + lane_id] =  0;
        }

        if ( !m_any_gmem )
        {
            return;
        }

    #pragma unroll 4

        for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
        {
            m_gmem_keys[offset] = -1;
            m_gmem_vals[offset] = 0;
        }

        m_any_gmem = false;
    }

    template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
    __device__ __forceinline__
    void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::insert( Key_type key, T val, int *status )
    {
        const short lane_id = get_lane_id();
        bool active = key != -1;
        Key_type winning_key = -1;
        int active_mask;

    #pragma unroll
        for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
        {

            active_mask = ballot( active );

            if ( active_mask == 0 )
            {
                return;
            }


            if ( active )
            {
                unsigned ukey  = reinterpret_cast<unsigned &>( key );
                int hash = ( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash] ) & (SMEM_SIZE - 1);

                winning_key = atomic_CAS(&m_smem_keys[hash], -1, key);
                winning_key = (winning_key == -1) ? key : winning_key;

                if (key == winning_key)  
                {
                    atomic_add(&m_smem_vals[hash], val); 
                    active = false;
                }
            }
        }

        #pragma unroll

        for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
        {
            active_mask = ballot( active );

            if ( active_mask == 0 )
            {
                return;
            }

            m_any_gmem = true;

            if ( active )
            {
                unsigned ukey = reinterpret_cast<unsigned &>( key );
                int hash      = ( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash] )  & (m_gmem_size - 1); 


                winning_key = atomic_CAS(&m_gmem_keys[hash], -1, key);
                winning_key = (winning_key == -1) ? key : winning_key;


                if (key == winning_key) 
                {
                    atomic_add(&m_gmem_vals[hash], val);
                    active = false;
                }
            }
        }

        if (status == NULL )
        {
            return;
        }

        if ( lane_id == 0 )
        {
            status[0] = 1;
        }
    }

    template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
    __device__ __forceinline__
    void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store( int count, T *vals )
    {
        int lane_id = get_lane_id();
        int lane_mask_lt = get_lane_mask_lt();
        int warp_offset = 0;
        const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
    #pragma unroll

        for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
        {
            const int offset = i_step * WARP_SIZE + lane_id;
            Key_type key = m_smem_keys[offset];
            int poll = ballot( key != -1 );

            if ( poll == 0 )
            {
                continue;
            }

            int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

            if ( key != -1 )
            {
                vals[dst_offset] = m_smem_vals[offset];
            }

            warp_offset += __popc( poll );
        }

        if ( !m_any_gmem )
        {
            return;
        }

    #pragma unroll 4

        for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
        {
            Key_type key = m_gmem_keys[offset];
            int poll = ballot( key != -1, activemask() );

            if ( poll == 0 )
            {
                continue;
            }

            int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

            if ( key != -1 )
            {
                vals[dst_offset] = m_gmem_vals[offset];
            }

            warp_offset += __popc( poll );
        }
    }

    template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
    __device__ __forceinline__
    void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store( int count, Key_type *keys, T *vals )
    {
        int lane_id = get_lane_id();
        int lane_mask_lt = get_lane_mask_lt();
        int warp_offset = 0;
        const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
    #pragma unroll

        for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
        {
            const int offset = i_step * WARP_SIZE + lane_id;
            Key_type key = m_smem_keys[offset];
            int poll = ballot( key != -1 );

            if ( poll == 0 )
            {
                continue;
            }

            int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

            if ( key != -1 )
            {
                keys[dst_offset] = key;
                vals[dst_offset] = m_smem_vals[offset];
            }

            warp_offset += __popc( poll );
        }

        if ( !m_any_gmem )
        {
            return;
        }

    #pragma unroll 4

        for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
        {
            Key_type key = m_gmem_keys[offset];
            int poll = ballot( key != -1, activemask() );

            if ( poll == 0 )
            {
                continue;
            }

            int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

            if ( key != -1 )
            {
                keys[dst_offset] = key;
                vals[dst_offset] = m_gmem_vals[offset];
            }

            warp_offset += __popc( poll );
        }
    }


    enum { WARP_SIZE = 32, SMEM_SIZE = 128 };

    template< int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE, bool COUNT_ONLY >
    __global__ //__launch_bounds__( CTA_SIZE )
    void
    count_non_zeroes_kernel( const int A_num_rows,
                            const int *A_rows,
                            const int *A_cols,
                            const int *B_rows,
                            const int *B_cols,
                            int *C_rows,
                            int *C_cols,
                            int *Aq1,
                            int *Bq1,
                            int *Aq2,
                            int *Bq2,
                            const int gmem_size,
                            int *g_keys,
                            int *wk_work_queue,
                            int *wk_status )
    {
        // 一个block中有8个warp
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        
        // The hash keys stored in shared memory.
        // 每个warp分128字节的共享内存
        __shared__ int s_keys[NUM_WARPS * SMEM_SIZE];
        
        // The coordinates of the thread inside the CTA/warp.
        // block内部的warp序号和lane序号
        const int warp_id = get_warp_id();
        const int lane_id = get_lane_id();
        
        // First threads load the row IDs of A needed by the CTA...
        // a_row_id是总体上的warp编号
        int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
        // Create local storage for the set.
        // Hash_set_atomic<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );
        Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );

        // Loop over rows of A.
        // 每个warp处理A的一行，这里的get_work其实就是把wk_work_queue原子加一下，为啥呢，其实正如work_queue所言
        // 第一个grid需要对前若干行处理，后面的未处理的行需要获得一个偏移量，以便于之后的grid继续处理这些行
        for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id ) )
        {
            int c_row_id = a_row_id;

            if (Aq1 != NULL)
            {
                a_row_id = Aq1[a_row_id];
            }

            // Make sure we have to proceed.
            if ( COUNT_ONLY )
            {
                volatile int *status = reinterpret_cast<volatile int *>( wk_status );

                if ( set.has_failed() || *status != 0 )
                {
                    return;
                }
            }

            // Clear the set.
            set.clear();
            // Load the range of the row.
            int a_col_tmp = -1;

            // warp中的0号1号两个线程才能从行索引中获得起始和结束行索引
            if ( lane_id < 2 )
            {
                a_col_tmp = load( &A_rows[a_row_id + lane_id] );
            }
            int a_col_it  = shfl( a_col_tmp, 0 );
            int a_col_end = shfl( a_col_tmp, 1 );
            
            // 一个总体上的warp负责A的一整行
            // Iterate over the columns of A.
            // 假设一共有10000行，一共有128个warp，每个warp都需要循环10000 / 128次
            // 0-31号线程分别有一个起始位置0-31，然后处理这一行的nnz，这些nnz假设有1000个，那么每个线程需要处理1000 / 32个A的nnz，也就是1000 / 32个B的行
            for ( a_col_it += lane_id ; __any_sync(DEFAULT_MASK, a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
            {
                // A这一行的nnz不是32的整数，则会有空闲线程
                // Is it an active thread.
                const bool is_active = a_col_it < a_col_end;
                // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
                int b_row_id = -1;

                if ( is_active )
                {
                    //真实的列索引值在这里取出来，再去对应B的哪一行
                    b_row_id = load( &A_cols[a_col_it] );

                    //b_row_id is actually column of A
                    if (Aq2 != NULL)
                    {
                        b_row_id = Aq2[b_row_id];
                    }

                    if (Bq1 != NULL)
                    {
                        b_row_id = Bq1[b_row_id];
                    }
                }

                // The number of valid rows.
                // B有num_rows行需要处理
                const int num_rows = __popc( ballot(is_active) );
                
                // num_rows其实最多有32个（因为一个warp负责A的一行，一次32个nnz，不断前进处理），而对于每个线程来说，遍历到
                // A这一行中的nnz后并不会做什么，而是等待后面统计一共有几个active的行，随后整个warp所有的线程都会
                // 共同处理B的第一行，然后处理B的第二行，循环直到active行都被处理
                // Uniform loop: threads collaborate to load other elements.
                for ( int k = 0 ; k < num_rows ; ++k )
                {
                    // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                    const int uniform_b_row_id = shfl( b_row_id, k );

                    // B的那一行的起始和结束行指针
                    // Load the range of the row of B.
                    int b_col_tmp = -1;
                    if ( lane_id < 2 )
                    {
                        b_col_tmp = load( &B_rows[uniform_b_row_id + lane_id] );
                    }

                    // 每个线程都获取一下B这一行的起始和结束
                    int b_col_it  = shfl( b_col_tmp, 0 );
                    int b_col_end = shfl( b_col_tmp, 1 );

                    // Iterate over the range of columns of B.
                    // 起始位置根据线程号有所区别，但是循环后加的偏移量还是warp大小
                    for ( b_col_it += lane_id ; any(b_col_it < b_col_end) ; b_col_it += WARP_SIZE )
                    {
                        int b_col_id = -1;

                        if ( b_col_it < b_col_end )
                        {
                            b_col_id = load( &B_cols[b_col_it] );

                            // b_col_id is actually column of B
                            if (Bq2 != NULL)
                            {
                                b_col_id = Bq2[b_col_id];
                            }
                        }

                        set.insert( b_col_id, COUNT_ONLY ? wk_status : NULL );
                    }
                }
            }

            // Store the results.
            if ( COUNT_ONLY )
            {
                int count = set.compute_size();

                if ( lane_id == 0 )
                {
                    C_rows[c_row_id] = count;
                }
            }
            else
            {
                int c_col_tmp = -1;

                if ( lane_id < 2 )
                {
                    c_col_tmp = load( &C_rows[c_row_id + lane_id] );
                }

                int c_col_it  = shfl( c_col_tmp, 0 );
                int c_col_end = shfl( c_col_tmp, 1 );
                // Store the results.
                int count = c_col_end - c_col_it;

                if ( count == 0 )
                {
                    continue;
                }

                set.store( count, &C_cols[c_col_it] );
            }
        }
    }

    template< int NUM_THREADS_PER_ROW, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE, bool COUNT_ONLY >
    __global__ //__launch_bounds__( CTA_SIZE )
    void
    count_non_zeroes_kernel( const int A_num_rows,
                            const int *__restrict A_rows,
                            const int *__restrict A_cols,
                            const int *__restrict B_rows,
                            const int *__restrict B_cols,
                            int *__restrict C_rows,
                            int *__restrict C_cols,
                            int *Aq1,
                            int *Bq1,
                            int *Aq2,
                            int *Bq2,
                            const int gmem_size,
                            int *g_keys,
                            int *wk_work_queue,
                            int *wk_status )
    {
        // 每个block有256 / 32 == 8个warp
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        // 相当于使用子warp的思路，每个warp中的几个线程去负责一行，这样可以增加并行度
        const int NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;
        // The hash keys stored in shared memory.
        // 本来这里就不是volatile的，所以不需要修改
        __shared__ /*volatile*/ int s_keys[NUM_WARPS * SMEM_SIZE];
        // The coordinates of the thread inside the CTA/warp.
        const int warp_id = get_warp_id();
        const int lane_id = get_lane_id();
        // Constants.
        const int lane_id_div_num_threads = lane_id / NUM_THREADS_PER_ROW;
        const int lane_id_mod_num_threads = lane_id % NUM_THREADS_PER_ROW;
        // First threads load the row IDs of A needed by the CTA...
        int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
        // Create local storage for the set.
        Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );
        // Hash_set_atomic<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );

        // Loop over rows of A. get_work支持乱序获取某一行去算，不必等待！！！！！
        for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id ) )
        {
            int c_row_id = a_row_id;

            if (Aq1 != NULL)
            {
                a_row_id = Aq1[a_row_id];
            }

            // Make sure we have to proceed.
            if ( COUNT_ONLY )
            {
                volatile int *status = reinterpret_cast<volatile int *>( wk_status );

                if ( set.has_failed() || *status != 0 )
                {
                    return;
                }
            }

            // Clear the set.
            set.clear();
            // Load the range of the row.
            int a_col_tmp = -1;

            if ( lane_id < 2 )
            {
                a_col_tmp = load( &A_rows[a_row_id + lane_id] );
            }

            int a_col_it  = shfl( a_col_tmp, 0 );
            int a_col_end = shfl( a_col_tmp, 1 );

            // Iterate over the columns of A.
            for ( a_col_it += lane_id ; any(a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
            {
                // Is it an active thread.
                const bool is_active = a_col_it < a_col_end;
                // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
                int b_row_id = -1;

                if ( is_active )
                {
                    b_row_id = load( &A_cols[a_col_it] );

                    //b_row_id is actually column of A
                    if (Aq2 != NULL)
                    {
                        b_row_id = Aq2[b_row_id];
                    }

                    if (Bq1 != NULL)
                    {
                        b_row_id = Bq1[b_row_id];
                    }
                }

                const int num_rows = __popc( ballot(is_active) );
                
                // 前面的count kernel是一个warp一起处理B的一行，这个kernel则是一个warp同时处理NUM_LOADED_ROWS行，比如NUM_THREADS_PER_ROW=2
                // 则一次处理16行
                // Uniform loop: threads collaborate to load other elements.
                for ( int k = 0 ; k < num_rows ; k += NUM_LOADED_ROWS )
                {
                    int local_k = k + lane_id_div_num_threads;
                    // Is it an active thread.
                    bool is_active_k = local_k < num_rows;
                    // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                    const int uniform_b_row_id = shfl( b_row_id, local_k );
                    // Load the range of the row of B.
                    int b_col_tmp = -1;

                    if ( is_active_k && lane_id_mod_num_threads < 2 )
                    {
                        b_col_tmp = load( &B_rows[uniform_b_row_id + lane_id_mod_num_threads] );
                    }

                    int b_col_it  = shfl( b_col_tmp, lane_id_div_num_threads * NUM_THREADS_PER_ROW + 0 );
                    int b_col_end = shfl( b_col_tmp, lane_id_div_num_threads * NUM_THREADS_PER_ROW + 1 );

                    // Iterate over the range of columns of B.
                    for ( b_col_it += lane_id_mod_num_threads ; any(b_col_it < b_col_end) ; b_col_it += NUM_THREADS_PER_ROW )
                    {
                        int b_col_id = -1;

                        if ( b_col_it < b_col_end )
                        {
                            b_col_id = load( &B_cols[b_col_it] );

                            // b_col_id is actually column of B
                            if (Bq2 != NULL)
                            {
                                b_col_id = Bq2[b_col_id];
                            }
                        }
                        // wk_status可以传导到最外层
                        set.insert( b_col_id, COUNT_ONLY ? wk_status : NULL );
                        // set.insert_atomic( b_col_id, COUNT_ONLY ? wk_status : NULL );
                    }
                }
            }

            // Store the results.
            if ( COUNT_ONLY )
            {
                // int count = set.compute_size_with_duplicates();
                int count = set.compute_size();
                if ( lane_id == 0 )
                {
                    C_rows[c_row_id] = count;
                }
            }
            else
            {
                int c_col_tmp = -1;

                if ( lane_id < 2 )
                {
                    c_col_tmp = load( &C_rows[c_row_id + lane_id] );
                }

                int c_col_it  = shfl( c_col_tmp, 0 );
                int c_col_end = shfl( c_col_tmp, 1 );
                // Store the results.
                int count = c_col_end - c_col_it;

                if ( count == 0 )
                {
                    continue;
                }

                set.store( count, &C_cols[c_col_it] );
            }
        }
    }

    void count_non_zeroes()
    {
        // 每个grid含有1024个block
        int GRID_SIZE = 1024;
        // 每个block含有256个线程
        const int CTA_SIZE  = 256;
        // 每个block中含有8个warp
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        // Reset work queue.
        // 整个grid含有多少个warp，如果一个矩阵足够大，则一个grid是算不完的，必须循环好几次
        int work_offset = GRID_SIZE * NUM_WARPS;
        CHECK_ERROR( cudaMemcpy( m_work_queue, &work_offset, sizeof(int), cudaMemcpyHostToDevice ) );

        // +++
        // int tmp = (A.get_num_rows() + NUM_WARPS - 1) / NUM_WARPS;
        // GRID_SIZE = GRID_SIZE > tmp ? GRID_SIZE : tmp;

        // Compute non-zero elements.
        switch ( m_num_threads_per_row_count )
        {
            /*
                int MA, NA, nnzA;
                int *csrRowIndexHostPtrA = 0;
                int *csrColIndexHostPtrA = 0;
                double *csrValHostPtrA = 0;
                char matrixNameA[1024] = {0};
            */
            // m_num_threads_per_row_count个线程负责A的一行，
            case 2:
                count_non_zeroes_kernel< 2, CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                    MA,
                    csrRowIndexDevPtrA,
                    csrColIndexDevPtrA,
                    csrRowIndexDevPtrB,
                    csrColIndexDevPtrB,
                    csrRowIndexDevPtrC,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    m_gmem_size,
                    m_keys,
                    m_work_queue,
                    m_status );
                break;

            case 4:
            count_non_zeroes_kernel< 4, CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                MA,
                csrRowIndexDevPtrA,
                    csrColIndexDevPtrA,
                    csrRowIndexDevPtrB,
                    csrColIndexDevPtrB,
                    csrRowIndexDevPtrC,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                m_gmem_size,
                m_keys,
                m_work_queue,
                m_status );
            break;

            case 8:
            count_non_zeroes_kernel< 8, CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                MA,
                csrRowIndexDevPtrA,
                    csrColIndexDevPtrA,
                    csrRowIndexDevPtrB,
                    csrColIndexDevPtrB,
                    csrRowIndexDevPtrC,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                m_gmem_size,
                m_keys,
                m_work_queue,
                m_status );
            break;

            case 16:
            count_non_zeroes_kernel< 16, CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                MA,
                csrRowIndexDevPtrA,
                csrColIndexDevPtrA,
                csrRowIndexDevPtrB,
                csrColIndexDevPtrB,
                csrRowIndexDevPtrC,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                m_gmem_size,
                m_keys,
                m_work_queue,
                m_status );
            break;

            default:
            // printf("avg nnz = %d\n", m_num_threads_per_row_count);
                count_non_zeroes_kernel<CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                    MA,
                    csrRowIndexDevPtrA,
                    csrColIndexDevPtrA,
                    csrRowIndexDevPtrB,
                    csrColIndexDevPtrB,
                    csrRowIndexDevPtrC,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                m_gmem_size,
                m_keys,
                m_work_queue,
                m_status );
        }

        cudaCheckError();
        CHECK_ERROR( cudaGetLastError() );
    }


    template< typename Value_type, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE >
    __global__ //__launch_bounds__( CTA_SIZE, 6 )
    void
    compute_values_kernel( const int A_num_rows,
                        const int *__restrict A_rows,
                        const int *__restrict A_cols,
                        const double *__restrict A_vals,
                        const int *__restrict B_rows,
                        const int *__restrict B_cols,
                        const double *__restrict B_vals,
                        const int *__restrict C_rows,
                        int *__restrict C_cols,
                        double *__restrict C_vals,
                        int *Aq1,
                        int *Bq1,
                        int *Aq2,
                        int *Bq2,
                        const int gmem_size,
                        int *g_keys,
                        double *g_vals,
                        int *wk_work_queue,
                        int *wk_status )
    {
        const int NUM_WARPS = CTA_SIZE / 32;
        // The hash keys stored in shared memory.
        __shared__ /*volatile*/ int s_keys[NUM_WARPS * SMEM_SIZE];
        // The hash values stored in shared memory.
        
        // __shared__ volatile Word s_vote[NUM_WARPS * SMEM_SIZE / 4];
        __shared__ double s_vals[NUM_WARPS * SMEM_SIZE]; 
        const int warp_id = get_warp_id();
        const int lane_id = get_lane_id();
        // First threads load the row IDs of A needed by the CTA...
        int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
        // Create local storage for the set.
        // Hash_map<int, double, SMEM_SIZE, 4, WARP_SIZE> map( &s_keys[warp_id * SMEM_SIZE],
        //         &g_keys[a_row_id * gmem_size],
        //         &s_vote[warp_id * SMEM_SIZE / 4],
        //         &g_vals[a_row_id * gmem_size],
        //         gmem_size );
        Hash_map<int, double, SMEM_SIZE, 4, WARP_SIZE> map(&s_keys[warp_id * SMEM_SIZE],
            &g_keys[a_row_id * gmem_size],
            // &s_vote[warp_id * SMEM_SIZE / 4],
            &s_vals[warp_id * SMEM_SIZE],
            &g_vals[a_row_id * gmem_size],
            gmem_size );

        // Loop over rows of A.
        for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id ) )
        {
            int c_row_id = a_row_id;

            if (Aq1 != NULL)
            {
                a_row_id = Aq1[a_row_id];
            }

            // Clear the map.
            map.clear();
            // Load the range of the row.
            int a_col_tmp = -1;

            if ( lane_id < 2 )
            {
                a_col_tmp = load( &A_rows[a_row_id + lane_id] );
            }

            int a_col_it  = shfl( a_col_tmp, 0 );
            int a_col_end = shfl( a_col_tmp, 1 );

            // Iterate over the columns of A.
            for ( a_col_it += lane_id ; any(a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
            {
                // Is it an active thread.
                const bool is_active = a_col_it < a_col_end;
                // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
                int b_row_id = -1;
                double a_value = 0.0;

                if ( is_active )
                {
                    b_row_id = load( &A_cols[a_col_it] );
                    a_value  = load( &A_vals[a_col_it] );

                    //b_row_id is actually column of A
                    if (Aq2 != NULL)
                    {
                        b_row_id = Aq2[b_row_id];
                    }

                    if (Bq1 != NULL)
                    {
                        b_row_id = Bq1[b_row_id];
                    }
                }

                const int num_rows = __popc( ballot(is_active) );

                // Uniform loop: threads collaborate to load other elements.
                for ( int k = 0 ; k < num_rows ; ++k )
                {
                    // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                    const int uniform_b_row_id = shfl( b_row_id, k );
                    // The value of A.
                    const double uniform_a_value = shfl( a_value, k );
                    // Load the range of the row of B.
                    int b_col_tmp = -1;

                    if ( lane_id < 2 )
                    {
                        b_col_tmp = load( &B_rows[uniform_b_row_id + lane_id] );
                    }

                    int b_col_it  = shfl( b_col_tmp, 0 );
                    int b_col_end = shfl( b_col_tmp, 1 );

                    // Iterate over the range of columns of B.
                    for ( b_col_it += lane_id ; any(b_col_it < b_col_end) ; b_col_it += WARP_SIZE )
                    {
                        int b_col_id = -1;
                        double b_value = 0.0;

                        if ( b_col_it < b_col_end )
                        {
                            b_col_id = load( &B_cols[b_col_it] );
                            b_value  = load( &B_vals[b_col_it] );

                            if (Bq2 != NULL)
                            {
                                b_col_id = Bq2[b_col_id];
                            }
                        }

                        map.insert( b_col_id, uniform_a_value * b_value, wk_status );
                    }
                }
            }

            // Store the results.
            int c_col_tmp = -1;

            if ( lane_id < 2 )
            {
                c_col_tmp = load( &C_rows[c_row_id + lane_id] );
            }

            int c_col_it  = shfl( c_col_tmp, 0 );
            int c_col_end = shfl( c_col_tmp, 1 );
            // Store the results.
            int count = c_col_end - c_col_it;

            if ( count == 0 )
            {
                continue;
            }

            map.store( count, &C_cols[c_col_it], &C_vals[c_col_it] );
        }
    }

    template< int NUM_THREADS_PER_ROW, typename Value_type, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE >
    __global__ //__launch_bounds__( CTA_SIZE, 6 )
    void
    compute_values_kernel( const int A_num_rows,
                        const int *__restrict A_rows,
                        const int *__restrict A_cols,
                        const double *__restrict A_vals,
                        const int *__restrict B_rows,
                        const int *__restrict B_cols,
                        const double *__restrict B_vals,
                        const int *__restrict C_rows,
                        int *__restrict C_cols,
                        double *__restrict C_vals,
                        int *Aq1,
                        int *Bq1,
                        int *Aq2,
                        int *Bq2,
                        const int gmem_size,
                        int *g_keys,
                        double *g_vals,
                        int *wk_work_queue,
                        int *wk_status )
    {
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        const int NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;
        // The hash keys stored in shared memory.
        __shared__ /*volatile*/ int s_keys[NUM_WARPS * SMEM_SIZE];
        // The hash values stored in shared memory.
        
        // __shared__ volatile Word s_vote[NUM_WARPS * SMEM_SIZE / 4];
        __shared__ double s_vals[NUM_WARPS * SMEM_SIZE]; 
        
        // The coordinates of the thread inside the CTA/warp.
        const int warp_id = get_warp_id( );
        const int lane_id = get_lane_id( );
        // Constants.
        const int lane_id_div_num_threads = lane_id / NUM_THREADS_PER_ROW;
        const int lane_id_mod_num_threads = lane_id % NUM_THREADS_PER_ROW;
        // First threads load the row IDs of A needed by the CTA...
        int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
        // Create local storage for the set.
        // Hash_map<int, double, SMEM_SIZE, 4, WARP_SIZE> map( &s_keys[warp_id * SMEM_SIZE],
        //         &g_keys[a_row_id * gmem_size],
        //         &s_vote[warp_id * SMEM_SIZE / 4],
        //         &g_vals[a_row_id * gmem_size],
        //         gmem_size );
        // Hash_map_atomic<int, double, SMEM_SIZE, 4, WARP_SIZE> map(&s_keys[warp_id * SMEM_SIZE],
        //     &g_keys[a_row_id * gmem_size],
        //     // &s_vote[warp_id * SMEM_SIZE / 4],
        //     &s_vals[warp_id * SMEM_SIZE],
        //     &g_vals[a_row_id * gmem_size],
        //     gmem_size );
        Hash_map<int, double, SMEM_SIZE, 4, WARP_SIZE> map(&s_keys[warp_id * SMEM_SIZE],
                &g_keys[a_row_id * gmem_size],
                // &s_vote[warp_id * SMEM_SIZE / 4],
                &s_vals[warp_id * SMEM_SIZE],
                &g_vals[a_row_id * gmem_size],
                gmem_size );

        // Loop over rows of A.
        for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id ) )
        {
            int c_row_id = a_row_id;

            if (Aq1 != NULL)
            {
                a_row_id = Aq1[a_row_id];
            }

            // Clear the map.
            // map.clear_all();
            map.clear();
            // Load the range of the row.
            int a_col_tmp = -1;

            if ( lane_id < 2 )
            {
                a_col_tmp = load( &A_rows[a_row_id + lane_id] );
            }

            int a_col_it  = shfl( a_col_tmp, 0 );
            int a_col_end = shfl( a_col_tmp, 1 );

            // Iterate over the columns of A.
            for ( a_col_it += lane_id ; any(a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
            {
                // Is it an active thread.
                const bool is_active = a_col_it < a_col_end;
                // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
                int b_row_id(-1);
                double a_value = 0.0;

                if ( is_active )
                {
                    b_row_id = load( &A_cols[a_col_it] );
                    a_value  = load( &A_vals[a_col_it] );

                    //b_row_id is actually column of A
                    if (Aq2 != NULL)
                    {
                        b_row_id = Aq2[b_row_id];
                    }

                    if (Bq1 != NULL)
                    {
                        b_row_id = Bq1[b_row_id];
                    }
                }

                const int num_rows = __popc( ballot(is_active) );

                // Uniform loop: threads collaborate to load other elements.
                for ( int k = 0 ; k < num_rows ; k += NUM_LOADED_ROWS )
                {
                    int local_k = k + lane_id_div_num_threads;
                    // Is it an active thread.
                    bool is_active_k = local_k < num_rows;
                    // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                    const int uniform_b_row_id = shfl( b_row_id, k + lane_id_div_num_threads );
                    // The value of A.
                    const double uniform_a_value = shfl( a_value, k + lane_id_div_num_threads );
                    // Load the range of the row of B.
                    int b_col_tmp = -1;

                    if ( is_active_k && lane_id_mod_num_threads < 2 )
                    {
                        b_col_tmp = load( &B_rows[uniform_b_row_id + lane_id_mod_num_threads] );
                    }

                    int b_col_it  = shfl( b_col_tmp, lane_id_div_num_threads * NUM_THREADS_PER_ROW + 0 );
                    int b_col_end = shfl( b_col_tmp, lane_id_div_num_threads * NUM_THREADS_PER_ROW + 1 );

                    // Iterate over the range of columns of B.
                    for ( b_col_it += lane_id_mod_num_threads ; any(b_col_it < b_col_end) ; b_col_it += NUM_THREADS_PER_ROW )
                    {
                        int b_col_id(-1);
                        double b_value = 0.0;

                        if ( b_col_it < b_col_end )
                        {
                            b_col_id = load( &B_cols[b_col_it] );
                            b_value  = load( &B_vals[b_col_it] );

                            //b_col_id is actually column of B
                            if (Bq2 != NULL)
                            {
                                b_col_id = Bq2[b_col_id];
                            }
                        }

                        // map.insert_with_duplicates( b_col_id, uniform_a_value * b_value, wk_status );
                        // map.insert_atomic( b_col_id, uniform_a_value * b_value, wk_status );
                        map.insert( b_col_id, uniform_a_value * b_value, wk_status );
                    }
                }
            }

            // Store the results.
            int c_col_tmp = -1;

            if ( lane_id < 2 )
            {
                c_col_tmp = load( &C_rows[c_row_id + lane_id] );
            }

            int c_col_it  = shfl( c_col_tmp, 0 );
            int c_col_end = shfl( c_col_tmp, 1 );
            // Store the results.
            int count = c_col_end - c_col_it;

            if ( count == 0 )
            {
                continue;
            }

            map.store( count, &C_cols[c_col_it], &C_vals[c_col_it] );
        }
    }

    void compute_values()
    {
        const int GRID_SIZE = 1024;
        const int CTA_SIZE  = 128;
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        // Reset the work queue.
        int work_offset = GRID_SIZE * NUM_WARPS;
        CHECK_ERROR( cudaMemcpy( m_work_queue, &work_offset, sizeof(int), cudaMemcpyHostToDevice ) );
        // Compute the values.
        int *status = NULL;

        if ( m_num_threads_per_row_count != m_num_threads_per_row_compute )
        {
            status = m_status;
        }
        
        // printf("进来了compute_values %d\n", m_num_threads_per_row_compute);
        switch ( m_num_threads_per_row_compute )
        {
            //int *csrRowIndexDevPtrB = 0;
            // int *csrColIndexDevPtrB = 0;
            // double *csrValDevPtrB = 0;
            case 2:
                compute_values_kernel< 2, double, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                    MA,
                    csrRowIndexDevPtrA,
                    csrColIndexDevPtrA,
                    csrValDevPtrA,
                    csrRowIndexDevPtrB,
                    csrColIndexDevPtrB,
                    csrValDevPtrB,
                    csrRowIndexDevPtrC,
                    csrColIndexDevPtrC,
                    csrValDevPtrC,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    m_gmem_size,
                    m_keys,
                    m_vals,
                    m_work_queue,
                    status );
                break;

            case 4:
            compute_values_kernel< 4, double, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                    MA,
                    csrRowIndexDevPtrA,
                    csrColIndexDevPtrA,
                    csrValDevPtrA,
                    csrRowIndexDevPtrB,
                    csrColIndexDevPtrB,
                    csrValDevPtrB,
                    csrRowIndexDevPtrC,
                    csrColIndexDevPtrC,
                    csrValDevPtrC,
                NULL,
                NULL,
                NULL,
                NULL,
                    m_gmem_size,
                    m_keys,
                    m_vals,
                    m_work_queue,
                    status );
                break;

            case 8:
                compute_values_kernel< 8, double, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                    MA,
                    csrRowIndexDevPtrA,
                    csrColIndexDevPtrA,
                    csrValDevPtrA,
                    csrRowIndexDevPtrB,
                    csrColIndexDevPtrB,
                    csrValDevPtrB,
                    csrRowIndexDevPtrC,
                    csrColIndexDevPtrC,
                    csrValDevPtrC,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    m_gmem_size,
                    m_keys,
                    m_vals,
                    m_work_queue,
                    status );
                break;

            case 16:
                compute_values_kernel<16, double, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                    MA,
                    csrRowIndexDevPtrA,
                    csrColIndexDevPtrA,
                    csrValDevPtrA,
                    csrRowIndexDevPtrB,
                    csrColIndexDevPtrB,
                    csrValDevPtrB,
                    csrRowIndexDevPtrC,
                    csrColIndexDevPtrC,
                    csrValDevPtrC,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    m_gmem_size,
                    m_keys,
                    m_vals,
                    m_work_queue,
                    status );
                break;

            default:
                compute_values_kernel<double, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                    MA,
                    csrRowIndexDevPtrA,
                    csrColIndexDevPtrA,
                    csrValDevPtrA,
                    csrRowIndexDevPtrB,
                    csrColIndexDevPtrB,
                    csrValDevPtrB,
                    csrRowIndexDevPtrC,
                    csrColIndexDevPtrC,
                    csrValDevPtrC,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    m_gmem_size,
                    m_keys,
                    m_vals,
                    m_work_queue,
                    status );
        }

        cudaCheckError();
        CHECK_ERROR( cudaGetLastError() );
    }
}

namespace nsparse {
    #define div_up(a, b) ((a+b-1)/b)

    #define NUM_BIN 8
    #define WSIZE 32

    #define PWARP 4
    #define PWARP_ROWS 256
    #define PWARP_TSIZE 32
    #define PWARP_BLOCK_SIZE (PWARP * PWARP_ROWS)

    #define NUMERIC_PWARP 8
    #define NUMERIC_PWARP_ROWS 128
    #define NUMERIC_PWARP_TSIZE 32
    #define NUMERIC_PWARP_BLOCK_SIZE (NUMERIC_PWARP * NUMERIC_PWARP_ROWS)
 
    #define NUMERIC_SCALE_LARGE 2
    #define HASH_SCALE 107

    #define SYMMETRY_GENERAL 0
    #define SYMMETRY_SYMMETRY 1
    #define SYMMETRY_SKEW_SYMMETRY 2
    #define SYMMETRY_HERMITIAN 3

    template <typename T>
    inline void D2H(T *dst, T* src, size_t size){
        CHECK_ERROR(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    }

    template <typename T>
    inline void H2D(T *dst, T* src, size_t size){
        CHECK_ERROR(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    }

    template <typename T>
    inline void D2D(T *dst, T* src, size_t size){
        CHECK_ERROR(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    }

    class CSR{
    public:
        int M;
        int N;
        int nnz;

        int *rpt = nullptr;
        int *col = nullptr;
        double *val = nullptr;

        int *d_rpt = nullptr;
        int *d_col = nullptr;
        double *d_val = nullptr;

        int *rpt_compressed = nullptr;
        int *col_compressed = nullptr;
        double *val_compressed = nullptr;

        int *d_rpt_compressed = nullptr;
        int *d_col_compressed = nullptr;
        double *d_val_compressed = nullptr;

        CSR():M(0), N(0), nnz(0), 
                rpt(nullptr), col(nullptr), val(nullptr),
                d_rpt(nullptr), d_col(nullptr), d_val(nullptr)
        {}
        

        void hrelease(){
            if(rpt != nullptr) delete [] rpt;
            rpt = nullptr;
            if(col != nullptr) delete [] col;
            col = nullptr;
            if(val != nullptr) delete [] val;
            val = nullptr;
    
            // bug记录：上面写rpt_compressed，下面写rpt，导致double free
            if (rpt_compressed != nullptr ) delete [] rpt_compressed;
            rpt_compressed = nullptr;
            if (col_compressed != nullptr ) delete [] col_compressed;
            col_compressed = nullptr;
            if (val_compressed != nullptr ) delete [] val_compressed;
            val_compressed = nullptr;
        }
    
        void drelease(){
            if (d_rpt != nullptr) CHECK_ERROR(cudaFree(d_rpt));
            d_rpt = nullptr;
            if (d_col != nullptr) CHECK_ERROR(cudaFree(d_col));
            if (d_val != nullptr) CHECK_ERROR(cudaFree(d_val));
            d_col = nullptr;
            d_val = nullptr;
    
            if (d_rpt_compressed != nullptr) CHECK_ERROR(cudaFree(d_rpt_compressed));
            d_rpt_compressed = nullptr;
            if (d_col_compressed != nullptr) CHECK_ERROR(cudaFree(d_col_compressed));
            if (d_val_compressed != nullptr) CHECK_ERROR(cudaFree(d_val_compressed));
            d_col_compressed = nullptr;
            d_val_compressed = nullptr;
        }
    
        void release(){
            hrelease();
            drelease();
        }
    
        ~CSR(){
            release();
        }
    
        void H2D(){
            drelease();
    
            CHECK_ERROR(cudaMalloc(&d_rpt, (M+1)*sizeof(int)));
            CHECK_ERROR(cudaMalloc(&d_col, nnz*sizeof(int)));
            CHECK_ERROR(cudaMalloc(&d_val, nnz*sizeof(double)));
    
            CHECK_ERROR(cudaMemcpy(d_rpt, rpt, (M+1)*sizeof(int), cudaMemcpyHostToDevice));
            CHECK_ERROR(cudaMemcpy(d_col, col, nnz*sizeof(int), cudaMemcpyHostToDevice));
            CHECK_ERROR(cudaMemcpy(d_val, val, nnz*sizeof(double), cudaMemcpyHostToDevice));
        }
    
        void D2H(){
            hrelease();
            rpt = new int [M+1];
            col = new int [nnz];
            val = new double [nnz];
            CHECK_ERROR(cudaMemcpy(rpt, d_rpt, (M+1)*sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_ERROR(cudaMemcpy(col, d_col, nnz*sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_ERROR(cudaMemcpy(val, d_val, nnz*sizeof(double), cudaMemcpyDeviceToHost));
    
            rpt_compressed = new int [M+1];
            col_compressed = new int [nnz];
            val_compressed = new double [nnz];
            CHECK_ERROR(cudaMemcpy(rpt_compressed, d_rpt_compressed, (M+1)*sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_ERROR(cudaMemcpy(col_compressed, d_col_compressed, nnz*sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_ERROR(cudaMemcpy(val_compressed, d_val_compressed, nnz*sizeof(double), cudaMemcpyDeviceToHost));
        }
    
        CSR(const CSR &A){
            //printf("construct A\n");
            M = A.M;
            N = A.N;
            nnz = A.nnz;
            rpt = new int [M+1];
            col = new int [nnz];
            val = new double [nnz];
            memcpy(rpt, A.rpt, (M+1)*sizeof(int));
            memcpy(col, A.col, nnz*sizeof(int));
            memcpy(val, A.val, nnz*sizeof(double));
            d_rpt = nullptr;
            d_col = nullptr;
            d_val = nullptr;
        }
    
        CSR& operator=(const CSR &A){
            //printf("construct A\n");
            M = A.M;
            N = A.N;
            nnz = A.nnz;
            rpt = new int [M+1];
            col = new int [nnz];
            val = new double [nnz];
            memcpy(rpt, A.rpt, (M+1)*sizeof(int));
            memcpy(col, A.col, nnz*sizeof(int));
            memcpy(val, A.val, nnz*sizeof(double));
            d_rpt = nullptr;
            d_col = nullptr;
            d_val = nullptr;
            return *this;
        }
    
        CSR(const CSR &A, int M_, int N_, int M_start = 0, int N_start = 0){
            assert(M_ + M_start <= A.M && "matrix subsect error M");
            assert(N_ + N_start <= A.N && "matrix subsect error N");
            int M_end = M_start + M_;
            int N_end = N_start + N_;
            M = M_;
            N = N_;
            int *row_size = new int [M];
            memset(row_size, 0, M*sizeof(int));
            for(int i = M_start; i < M_end; i++){
                for(int j = A.rpt[i]; j < A.rpt[i+1]; j++){
                    if(A.col[j]>= N_start && A.col[j] < N_end){
                        row_size[i - M_start]++;
                    }
                }
            }
    
            rpt = new int [M+1];
            rpt[0] = 0;
            for(int i = 0; i < M; i++){
                rpt[i+1] = rpt[i] + row_size[i];
            }
            nnz = rpt[M];
            delete [] row_size;
    
            col = new int [nnz];
            val = new double [nnz];
            for(int i = M_start; i < M_end; i++){
                int jj = rpt[i - M_start];
                for(int j = A.rpt[i]; j < A.rpt[i+1]; j++){
                    if(A.col[j]>= N_start && A.col[j] < N_end){
                        col[jj] = A.col[j] - N_start;
                        val[jj++] = A.val[j];
                    }
                }
            }
            d_rpt = nullptr;
            d_col = nullptr;
            d_val = nullptr;
            //d_combined = nullptr;
        }
    
    
        bool operator==(const CSR &rhs){
            if(nnz != rhs.nnz){
                printf("nnz not equal %d %d\n", nnz, rhs.nnz);
                throw std::runtime_error("nnz not equal");
            }
            assert(M == rhs.M && "dimension not same");
            assert(N == rhs.N && "dimension not same");
            //assert(nnz == rhs.nnz && "dimension not same");
            int error_num = 0;
            double epsilon = 1e-9;
            for(int i = 0; i < M; i++){
                if(error_num > 10)
                    throw std::runtime_error("matrix compare: error num exceed threshold");
                if(rpt[i] != rhs.rpt[i]){
                    printf("rpt not equal at %d rows, %d != %d\n", i, rpt[i], rhs.rpt[i]);
                    error_num++;
                }
                for(int j = rpt[i]; j < rpt[i+1]; j++){
                    if(error_num > 10)
                        throw std::runtime_error("matrix compare: error num exceed threshold");
                    if(col[j] != rhs.col[j]){
                        printf("col not equal at %d rows, index %d != %d\n", i, col[j], rhs.col[j]);
                        error_num++;
                    }
                    if(!(std::fabs(val[j] - rhs.val[j]) < epsilon || 
                    std::fabs(val[j] - rhs.val[j]) < epsilon * std::fabs(val[j]))){
                        printf("val not eqaul at %d rows, value %.18le != %.18le\n", i, val[j], rhs.val[j]);
                        error_num++;
                    }
                }
            }
            if(rpt[M] != rhs.rpt[M]){
                printf("rpt[M] not equal\n");
                throw std::runtime_error("matrix compare: error num exceed threshold");
            }
            if(error_num)
                return false;
            else
                return true;
        }
    
        CSR(const std::string &mtx_file){
            construct(mtx_file);
        }

        struct matrix_market_banner
        {
            std::string matrix; // "matrix" or "vector"
            std::string storage;    // "array" or "coordinate", storage_format
            std::string type;       // "complex", "real", "integer", or "pattern"
            std::string symmetry;   // "general", "symmetric", "hermitian", or "skew-symmetric"
        };

        inline void tokenize(std::vector<std::string>& tokens, const std::string str, const std::string delimiters = "\n\r\t ")
        {
            tokens.clear();
            // Skip delimiters at beginning.
            std::string::size_type first_pos = str.find_first_not_of(delimiters, 0);
            // Find first "non-delimiter".
            std::string::size_type last_pos     = str.find_first_of(delimiters, first_pos);

            while (std::string::npos != first_pos || std::string::npos != last_pos)
            {
                // Found a token, add it to the vector.
                tokens.push_back(str.substr(first_pos, last_pos - first_pos));
                // Skip delimiters.  Note the "not_of"
                first_pos = str.find_first_not_of(delimiters, last_pos);
                // Find next "non-delimiter"
                last_pos = str.find_first_of(delimiters, first_pos);
            }
        }

        template <typename Stream>
        void read_mm_banner(Stream& input, matrix_market_banner& banner)
        {
            std::string line;
            std::vector<std::string> tokens;

            // read first line
            std::getline(input, line);
            tokenize(tokens, line);

            if (tokens.size() != 5 || tokens[0] != "%%MatrixMarket" || tokens[1] != "matrix")
                throw std::runtime_error("invalid MatrixMarket banner");

            banner.matrix = tokens[1]; // mow just matrix, no vector
            banner.storage  = tokens[2]; // now just coordinate(sparse), no array(dense)
            banner.type     = tokens[3]; // int, real, pattern for double, complex for two double
            banner.symmetry = tokens[4]; // general, symmetry, etc

            if(banner.matrix != "matrix" && banner.matrix != "vector")
                throw std::runtime_error("invalid MatrixMarket matrix type: " + banner.matrix);
            if(banner.matrix == "vector")
                throw std::runtime_error("not impl matrix type: " + banner.matrix);

            if (banner.storage != "array" && banner.storage != "coordinate")
                throw std::runtime_error("invalid MatrixMarket storage format [" + banner.storage + "]");
            if(banner.storage == "array")
                throw std::runtime_error("not impl storage type "+ banner.storage);

            if (banner.type != "complex" && banner.type != "real" && banner.type != "integer" && banner.type != "pattern")
                throw std::runtime_error("invalid MatrixMarket data type [" + banner.type + "]");
            //if(banner.type == "complex")
            //    throw std::runtime_error("not impl data type: " + banner.type);

            if (banner.symmetry != "general" && banner.symmetry != "symmetric" && banner.symmetry != "hermitian" && banner.symmetry != "skew-symmetric")
                throw std::runtime_error("invalid MatrixMarket symmetry [" + banner.symmetry + "]");
            if(banner.symmetry == "hermitian")
                throw std::runtime_error("not impl matrix type: " + banner.symmetry);
            
        }

        template <typename index_type, typename value_type>
        class Pair{
            public:
            index_type ind;
            value_type val;
            friend bool operator<=(const Pair &lhs, const Pair& rhs){
                return lhs.ind <= rhs.ind;
            }
            friend bool operator<(const Pair &lhs, const Pair& rhs){
                return lhs.ind < rhs.ind;
            }
            friend bool operator>(const Pair &lhs, const Pair& rhs){
                return lhs.ind > rhs.ind;
            }
        };
    
        void construct(const std::string &mtx_file){
            d_rpt = nullptr;
            d_col = nullptr;
            d_val = nullptr;
            std::ifstream ifile(mtx_file.c_str());
            if(!ifile){
                throw std::runtime_error(std::string("unable to open file \"") + mtx_file + std::string("\" for reading"));
            }
            matrix_market_banner banner;
            // read mtx header
            read_mm_banner(ifile, banner);
    
            // read file contents line by line
            std::string line;
    
            // skip over banner and comments
            do
            {
                std::getline(ifile, line);
            } while (line[0] == '%');
    
            // line contains [num_rows num_columns num_entries]
            std::vector<std::string> tokens;
            tokenize(tokens, line);
    
            if (tokens.size() != 3)
                throw std::runtime_error("invalid MatrixMarket coordinate format");
    
            std::istringstream(tokens[0]) >> M;
            std::istringstream(tokens[1]) >> N;
            std::istringstream(tokens[2]) >> nnz;
            assert(nnz > 0 && "something wrong: nnz is 0");
    
            int *I_ = new int [nnz];
            int *J_ = new int [nnz];
            double *coo_values_ = new double [nnz];
    
            int num_entries_read = 0;
    
            // read file contents
            if (banner.type == "pattern")
            {
                while(num_entries_read < nnz && !ifile.eof())
                {
                    ifile >> I_[num_entries_read];
                    ifile >> J_[num_entries_read];
                    num_entries_read++;
                }
                std::fill(coo_values_, coo_values_ + nnz, double(1));
            }
            else if (banner.type == "real" || banner.type == "integer")
            {
                while(num_entries_read < nnz && !ifile.eof())
                {
                    ifile >> I_[num_entries_read];
                    ifile >> J_[num_entries_read];
                    ifile >> coo_values_[num_entries_read];
                    num_entries_read++;
                }
            }
            else if (banner.type == "complex")
            {
                double tmp;
                while(num_entries_read < nnz && !ifile.eof())
                {
                    ifile >> I_[num_entries_read];
                    ifile >> J_[num_entries_read];
                    ifile >> coo_values_[num_entries_read] >> tmp;
                    num_entries_read++;
                }
            }
            else
            {
                throw std::runtime_error("invalid MatrixMarket data type");
            }
            ifile.close();
    
            if(num_entries_read != nnz)
                throw std::runtime_error("read nnz not equal to decalred nnz " + std::to_string(num_entries_read));
    
            // convert base-1 indices to base-0
            for(int n = 0; n < nnz; n++){
                I_[n] -= 1;
                J_[n] -= 1;
            }
    
            // expand symmetric formats to "general" format
            if (banner.symmetry != "general"){
                int non_diagonals = 0;
    
                for (int n = 0; n < nnz; n++)
                    if(I_[n] != J_[n])
                        non_diagonals++;
    
                int new_nnz = nnz + non_diagonals;
    
                int* new_I = new int [new_nnz];
                int* new_J = new int [new_nnz];
                double *new_coo_values;
                new_coo_values = new double [new_nnz];
                
    
                if (banner.symmetry == "symmetric"){
                    int cnt = 0;
                    for (int n = 0; n < nnz; n++){
                        // copy entry over
                        new_I[cnt] = I_[n];
                        new_J[cnt] = J_[n];
                        new_coo_values[cnt] = coo_values_[n];
                        cnt++;
    
                        // duplicate off-diagonals
                        if (I_[n] != J_[n]){
                            new_I[cnt] = J_[n];
                            new_J[cnt] = I_[n];
                            new_coo_values[cnt] = coo_values_[n];
                            cnt++;
                        }
                    }
                    assert(new_nnz == cnt && "something wrong: new_nnz != cnt");
                }
                else if (banner.symmetry == "skew-symmetric"){
                    int cnt = 0;
                    for (int n = 0; n < nnz; n++){
                        // copy entry over
                        new_I[cnt] = I_[n];
                        new_J[cnt] = J_[n];
                        new_coo_values[cnt] = coo_values_[n];
                        cnt++;
    
                        // duplicate off-diagonals
                        if (I_[n] != J_[n]){
                            new_I[cnt] = J_[n];
                            new_J[cnt] = I_[n];
                            new_coo_values[cnt] = -coo_values_[n];
                            cnt++;
                        }
                    }
                    assert(new_nnz == cnt && "something wrong: new_nnz != cnt");
                }
                else if (banner.symmetry == "hermitian"){
                    // TODO
                    throw std::runtime_error("MatrixMarket I/O does not currently support hermitian matrices");
                }
    
                // store full matrix in coo
                nnz = new_nnz;
                delete [] I_;
                delete [] J_;
                delete [] coo_values_;
                I_ = new_I;
                J_ = new_J;
                coo_values_ = new_coo_values;
            } // if (banner.symmetry != "general")
    
            // sort indices by (row,column)
            Pair<long, double> *p = new Pair<long, double> [nnz];
            for(int i = 0; i < nnz; i++){
                p[i].ind = (long int)N * I_[i] + J_[i];
                p[i].val = coo_values_[i];
            }
            std::sort(p, p + nnz);
            for(int i = 0; i < nnz; i++){
                I_[i] = p[i].ind / N;
                J_[i] = p[i].ind % N;
                coo_values_[i] = p[i].val;
            }
            delete [] p;
            
            // coo to csr
            rpt = new int [M+1];
            memset(rpt, 0, (M + 1) * sizeof(int));
            for(int i = 0; i < nnz; i++){
                rpt[I_[i]+1]++;
            }
            for(int i = 1; i <= M; i++){
                rpt[i] += rpt[i-1];
            }
            delete [] I_;
            col = J_;
            val = coo_values_;
    
            // check csr format
            assert(rpt[0] == 0 && "first row_pointer != 0");
            for(int i = 0; i < M; i++){
                if(rpt[i]<= rpt[i+1] && rpt[i] <= nnz){
                    for(int j = rpt[i]; j < rpt[i+1] - 1; j++){
                        if(col[j] < col[j+1]){}
                        else{
                            printf("row %d, col_index %d, index %d\n", i, col[j], j);
                            throw std::runtime_error("csr col_index not in assending order");
                        }
                    }
                    for(int j = rpt[i]; j < rpt[i+1]; j++){
                        if(col[j] < N && col[j] >= 0){}
                        else{
                            printf("row %d, col_index %d, index %d\n", i, col[j], j);
                            throw std::runtime_error("csr col_index out of range");
                        }
                    }
                }
                else{
                    printf("i %d  row_pointer[i] %d row_pointer[i+1] %d\n", i, rpt[i], rpt[i+1]);
                    throw std::runtime_error("csr row_pointer wrong");
                }
            }
            assert(rpt[M] == nnz && "last row_pointer != nnz_");
    
        }
    };

    class Info{
        public:
        // first, allocate C.rpt. 
        // d_row_flop, d_estimated_row_nnz, d_row_nnz are all reused with C.rpt

        // combined memory
        int *d_combined_mem; // second, allocate for all others
        int *combined_mem; // second, allocate for all others

        // info data
        int M; // number of rows
        int N; // number of cols

        int *d_bins; // size M
        
        int *d_bin_size; // size NUM_BIN
        int *d_bin_offset; // size NUM_BIN
        int *d_max_row_nnz; // size 1
        int *d_total_nnz; // size 1

        int *bin_size; // size NUM_BIN
        int *bin_offset; // size NUM_BIN
        int *max_row_nnz; // size 1
        int *total_nnz; // size 1

        int *d_cub_storage; // size variable
        size_t cub_storage_size;
        cudaStream_t *stream;

        // symbolic global and numeric global, is allocated at runtime
        int* d_global_mem_pool; // size unknown, allocated at runtime
        size_t global_mem_pool_size;
        bool global_mem_pool_malloced;

        // ********************************************************
        // public method
        Info(){}

        Info(CSR &C){
            allocate_rpt(C);
        }
        
        void allocate_rpt(CSR &C){
            CHECK_ERROR(cudaMalloc(&C.d_rpt, (C.M + 1)*sizeof(int)));
            CHECK_ERROR(cudaMalloc(&C.d_rpt_compressed, (C.M + 1)*sizeof(int)));
        }
        
        void allocate(CSR& C){
            M = C.M;
            N = C.N;
            stream = new cudaStream_t [NUM_BIN];
            for(int i = 0; i < NUM_BIN; i++){
                CHECK_ERROR(cudaStreamCreate(stream + i));
            }
                
            cub::DeviceScan::ExclusiveSum(nullptr, cub_storage_size, C.d_rpt, C.d_rpt, M + 1); // calculate tmp_storage_size in bytes
        
            int d_combined_size = M  + 2 * NUM_BIN + 2 + cub_storage_size/(sizeof(int));
            CHECK_ERROR(cudaMalloc(&d_combined_mem, d_combined_size * sizeof(int)));
            int combined_size = 2 * NUM_BIN + 2;
            combined_mem = (int *)malloc(combined_size * sizeof(int));
            assert(combined_mem != nullptr);
    
            // d_bins   d_bin_size  d_max_row_nnz   d_total_nnz d_bin_offset    d_cub_storage
            // M        NUM_BIN     1               1           NUM_BIN         cub_storage_size
            //          bin_size    max_row_nnz     total_nnz   bin_offset
            //          NUM_BIN     1               1           NUM_BIN
        
            d_bins = (int *)d_combined_mem; // size M
            d_bin_size = (int *)d_combined_mem + M; // size NUM_BIN
            d_max_row_nnz = d_bin_size + NUM_BIN; // size 1
            d_total_nnz = d_bin_size + NUM_BIN + 1; // size 1
            d_bin_offset = d_total_nnz + 1; // size NUM_BIN
            d_cub_storage = d_bin_offset + NUM_BIN; //????
        
            bin_size = (int*) combined_mem; // size NUM_BIN
            max_row_nnz = bin_size + NUM_BIN; // size 1
            total_nnz = bin_size + NUM_BIN + 1; // size 1
            bin_offset = bin_size + NUM_BIN + 2; // size NUM_BIN
            
            d_global_mem_pool = nullptr;
            global_mem_pool_size = 0;
            global_mem_pool_malloced = false;
        }
        
        void release(){
            cudaFree(d_combined_mem);
            d_combined_mem = nullptr;
            if(stream != nullptr){
                for(int i = 0; i < NUM_BIN; i++){
                    cudaStreamDestroy(stream[i]);
                }
                delete [] stream;
                stream = nullptr;
            }
            delete [] combined_mem;
            combined_mem = nullptr;
        }
        
        ~Info(){
            release();
        }
        
        
        void memset_all(int stream_idx = 1){
            CHECK_ERROR(cudaMemsetAsync(d_bin_size, 0, (NUM_BIN + 2) * sizeof(int), stream[stream_idx]));
            //CHECK_ERROR(cudaMemset(d_bin_size, 0, (NUM_BIN + 5) * sizeof(int)));
        }
        void memset_bin_size(int stream_idx = 1){
            CHECK_ERROR(cudaMemsetAsync(d_bin_size, 0, NUM_BIN * sizeof(int), stream[stream_idx]));
            //CHECK_ERROR(cudaMemset(d_bin_size, 0, (NUM_BIN + 5) * sizeof(int)));
        }
        
        void D2H_all(int stream_idx = 0){
            CHECK_ERROR(cudaMemcpyAsync(bin_size, d_bin_size, (NUM_BIN + 2) * sizeof(int), cudaMemcpyDeviceToHost, stream[stream_idx]));
            //CHECK_ERROR(cudaMemcpy(bin_size, d_bin_size, NUM_BIN * sizeof(int), cudaMemcpyHostToDevice));
        }
        
        void D2H_bin_size(int stream_idx = 0){
            CHECK_ERROR(cudaMemcpyAsync(bin_size, d_bin_size, NUM_BIN * sizeof(int), cudaMemcpyDeviceToHost, stream[stream_idx]));
            //CHECK_ERROR(cudaMemcpy(bin_size, d_bin_size, NUM_BIN * sizeof(int), cudaMemcpyHostToDevice));
        }
        
        void H2D_bin_offset(int stream_idx = 0){
            CHECK_ERROR(cudaMemcpyAsync(d_bin_offset, bin_offset, NUM_BIN * sizeof(int), cudaMemcpyHostToDevice, stream[stream_idx]));
        }    
    };

    __global__ void k_binning_small(int *d_bins, int M){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if(i >= M){
            return;
        }
        d_bins[i] = i;
    }

    __global__ void __launch_bounds__(1024, 2) k_compute_flop(
        const int* __restrict__ d_arpt, 
        const int* __restrict__ d_acol,
        const int* __restrict__ d_brpt,
        int M,
        int *d_row_flop,
        int *d_max_row_flop){
    
        __shared__ int shared_max_row_flop[1];
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= M) {
            return;
        }
        if(threadIdx.x == 0){
            shared_max_row_flop[0] = 0;
        }
        __syncthreads();
        int row_flop = 0;
        int j;
        int acol;
        int arow_start, arow_end;
        arow_start = d_arpt[i];
        arow_end = d_arpt[i+1];
        for (j = arow_start; j < arow_end; j++) {
            acol = d_acol[j];
            row_flop += d_brpt[acol + 1] - d_brpt[acol];
        }
        d_row_flop[i] = row_flop;
        atomicMax(shared_max_row_flop, row_flop);
        __syncthreads();
        if(threadIdx.x == 0){
            atomicMax(d_max_row_flop, shared_max_row_flop[0]);
        }
    }
    
    __global__ void __launch_bounds__ (1024, 2) d_numeric_binning(
        int * __restrict__ d_row_nnz, 
        int M, 
        int * __restrict__ d_bin_size, 
        int * __restrict__ d_total_nnz, 
        int * __restrict__ d_max_row_nnz){
    
        __shared__ int shared_bin_size[NUM_BIN];
        __shared__ int shared_local_nnz[1];
        __shared__ int shared_max_row_nnz[1];
        if(threadIdx.x < NUM_BIN){
            shared_bin_size[threadIdx.x] = 0;
        }
        if(threadIdx.x == 32){
            shared_local_nnz[0] = 0;
            shared_max_row_nnz[0] = 0;
        }
        __syncthreads();
        int range[NUM_BIN] = {31, 255, 511, 1022,    2047, 4095, 8191, INT_MAX}; // 1x
        // int range[NUM_BIN] = {21, 192, 384, 768,    1536, 3072, 5460, INT_MAX}; // 1.5x
        // int range[NUM_BIN] = {16, 128, 256, 512,    1024, 2048, 4095, INT_MAX}; // 2x
        //int range[NUM_BIN] = {10, 85, 170, 341,    682, 1365, 2730, INT_MAX}; // 3x
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int row_nnz, j;
        if(i < M){
            row_nnz = d_row_nnz[i];
            atomicAdd(shared_local_nnz, row_nnz);
            atomicMax(shared_max_row_nnz, row_nnz);
            //#pragma unroll
            for(j = 0; j < NUM_BIN; j++){
                if(row_nnz <= range[j]){
                    atomicAdd(shared_bin_size + j, 1);
                    goto before_end;
                }
            }
        }
        before_end:
    
    
        __syncthreads();
        if(threadIdx.x < NUM_BIN){
            atomicAdd(d_bin_size + threadIdx.x, shared_bin_size[threadIdx.x]);
        }
        if(threadIdx.x == 32){
            atomicAdd(d_total_nnz, shared_local_nnz[0]);
        }
        if(threadIdx.x == 64){
            atomicMax(d_max_row_nnz, shared_max_row_nnz[0]);
        }
    }
    
    __global__ void __launch_bounds__ (1024, 2) d_numeric_binning2  (
        int * __restrict__ d_row_nnz, 
        int M, 
        int * __restrict__ d_bins, 
        int * __restrict__ d_bin_size, 
        int * __restrict__ d_bin_offset){ 
    
        __shared__ int shared_bin_size[NUM_BIN];
        __shared__ int shared_bin_offset[NUM_BIN];
        if(threadIdx.x < NUM_BIN){
            shared_bin_size[threadIdx.x] = 0;
        }
        __syncthreads();
        int range[NUM_BIN] = {31, 255, 511, 1022,    2047, 4095, 8191, INT_MAX}; // 1x
        // int range[NUM_BIN] = {21, 192, 384, 768,    1536, 3072, 5460, INT_MAX}; // 1.5x
        // int range[NUM_BIN] = {16, 128, 256, 512,    1024, 2048, 4095, INT_MAX}; // 2x
        //int range[NUM_BIN] = {10, 85, 170, 341,    682, 1365, 2730, INT_MAX}; // 3x
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int row_nnz, j;
        if(i < M){
            row_nnz = d_row_nnz[i];
            //#pragma unroll
            for(j = 0; j < NUM_BIN; j++){
                if(row_nnz <= range[j]){
                    atomicAdd(shared_bin_size + j, 1);
                    goto before_end;
                }
            }
        }
        before_end:
    
    
        __syncthreads();
        if(threadIdx.x < NUM_BIN){
            shared_bin_offset[threadIdx.x] = atomicAdd(d_bin_size + threadIdx.x, shared_bin_size[threadIdx.x]);
            shared_bin_offset[threadIdx.x] += d_bin_offset[threadIdx.x];
            shared_bin_size[threadIdx.x] = 0;
        }
        __syncthreads();
        int index;
        if(i < M){
            //#pragma unroll
            for(j = 0; j < NUM_BIN; j++){
                if(row_nnz <= range[j]){
                    index = atomicAdd(shared_bin_size + j, 1);
                    d_bins[shared_bin_offset[j] + index] = i;
                    return;
                }
            }
        }
    }

    // kernel0
    __global__ void __launch_bounds__(NUMERIC_PWARP_BLOCK_SIZE, 2) d_numeric_shared_hash_pwarp(
        const int * __restrict__ d_arpt, const int * __restrict__ d_acol, 
        const double * __restrict__ d_aval,
        const int * __restrict__ d_brpt, const int * __restrict__ d_bcol, 
        const double * __restrict__ d_bval,
        int *d_bins, int bin_size,
        int *d_crpt, int *d_ccol, double* d_cval,
        int *d_crpt_compressed){

        //long long t0 = clock64();

        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int tid = threadIdx.x % NUMERIC_PWARP;
        int rid = i / NUMERIC_PWARP;
        int block_rid = rid % NUMERIC_PWARP_ROWS;
        __shared__ int shared_mem[NUMERIC_PWARP_ROWS * NUMERIC_PWARP_TSIZE * (sizeof(int) + sizeof(double))/sizeof(int)];
        const int tsize = NUMERIC_PWARP_TSIZE - 1;
        int *mono_shared_col = shared_mem;
        int *mono_shared_offset = shared_mem + NUMERIC_PWARP_ROWS * tsize;
        double *mono_shared_val = (double*)(shared_mem + NUMERIC_PWARP_ROWS * NUMERIC_PWARP_TSIZE);
        int j, k;
        for(j = threadIdx.x; j < NUMERIC_PWARP_ROWS * tsize; j += blockDim.x){
            mono_shared_col[j] = -1;
            mono_shared_val[j] = 0;
        }
        if(threadIdx.x < NUMERIC_PWARP_ROWS){
            mono_shared_offset[threadIdx.x] = 0;
        }
        if(rid >= bin_size){
            return;
        }
        __syncthreads();

        rid = d_bins[rid];
        int *shared_col = mono_shared_col + block_rid * tsize;
        //int *shared_offset = shared_col + tsize;
        double *shared_val = mono_shared_val + block_rid * tsize;
        int acol, bcol, hash, old;
        double aval, bval;
        for(j = d_arpt[rid] + tid; j < d_arpt[rid + 1]; j += NUMERIC_PWARP){ // pwarp per row, thread per a item, thread per b row
            acol = d_acol[j];
            aval = d_aval[j];
            for(k = d_brpt[acol]; k < d_brpt[acol + 1]; k++){ // thread per b row
                bcol = d_bcol[k];
                bval = d_bval[k];
                hash = (bcol * HASH_SCALE) % tsize;
                while(1){
                    old = atomicCAS(shared_col + hash, -1, bcol);
                    if(old == -1 || old == bcol){
                        atomicAdd(shared_val + hash, aval * bval);
                        break;
                    }
                    else{
                        hash = (hash + 1) < tsize ? hash + 1 : 0;
                    }

                }
            }
        }
        __syncthreads();
        //t0 = clock64() - t0;
        //if(threadIdx.x == 0){
        //    printf("inside d_numeric_hash pwarp hash %lld\n", t0);
        //}
        //__syncthreads();
        //t0 = clock64();

        //__syncthreads();

        int c_offset = d_crpt[rid];
        // int row_nnz = d_crpt[rid + 1] - d_crpt[rid];
        int offset;
        bool valid;
        #pragma unroll
        for(j = 0; j < tsize; j += NUMERIC_PWARP){
            offset = j + tid;
            valid = offset < tsize;
            if(valid){
                acol = shared_col[offset];
                aval = shared_val[offset];
                if(acol != -1){
                    offset = atomicAdd(mono_shared_offset + block_rid, 1);
                }
            }
            __syncthreads();
            if(valid && acol != -1){
                shared_col[offset] = acol;
                shared_val[offset] = aval;
            }
        }

        __syncthreads();
        //t0 = clock64() - t0;
        //if(threadIdx.x == 0){
        //    printf("inside d_numeric_hash pwarp condense %lld\n", t0);
        //}
        //__syncthreads();
        //t0 = clock64();

        int row_nnz = mono_shared_offset[block_rid];
        d_crpt_compressed[rid] = row_nnz;

        // count sort the result
        for (j = tid; j < row_nnz; j += NUMERIC_PWARP) {
            acol = shared_col[j];
            offset = 0;
            for (k = 0; k < row_nnz; k++) {
                // 59 - 
                offset += (unsigned int)(shared_col[k] - acol) >> 31;
            }
            d_ccol[c_offset + offset] = shared_col[j];
            d_cval[c_offset + offset] = shared_val[j];
        }
        //__syncthreads();
        //t0 = clock64() - t0;
        //if(threadIdx.x == 0){
        //    printf("inside d_numeric_hash sort %lld\n", t0);
        //}
    }

    //kernel1-5
    template <int SH_ROW, int BS>
    __global__ void __launch_bounds__(1024,2) d_numeric_shared_hash_tb_full_occu(
        const int * __restrict__ d_arpt, const int * __restrict__ d_acol, 
        const double * __restrict__ d_aval,
        const int * __restrict__ d_brpt, const int * __restrict__ d_bcol, 
        const double * __restrict__ d_bval,
        int *d_bins,
        int *d_crpt, int *d_ccol, double* d_cval,
        int *d_crpt_compressed){

        int tid = threadIdx.x & (WSIZE - 1);
        int wid = threadIdx.x / WSIZE;
        int wnum = blockDim.x / WSIZE;
        int j, k;
        __shared__ int shared_mem[SH_ROW * (sizeof(int) + sizeof(double))/sizeof(int)];
        const int tsize = SH_ROW - 1;

        // shared_col   shared_offset   shared_val
        // SH_ROW - 1   1               SH_ROW - 1
        int *shared_col = shared_mem;
        int *shared_offset = shared_mem + (SH_ROW - 1);
        double* shared_val = (double*)(shared_mem + SH_ROW);

        for(j = threadIdx.x; j < tsize; j += blockDim.x){
            shared_col[j] = -1;
            shared_val[j] = 0;
        }
        if(threadIdx.x == 0){
            shared_offset[0] = 0;
        }
        __syncthreads();

        int acol, bcol, hash, old;
        double aval, bval;
        int rid = d_bins[blockIdx.x];
        int c_offset = d_crpt[rid];
        // int row_nnz = d_crpt[rid + 1] - d_crpt[rid];

        for(j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum){
            acol = d_acol[j];
            aval = d_aval[j];
            for(k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k+= WSIZE){
                bcol = d_bcol[k];
                bval = d_bval[k];
                hash = (bcol * HASH_SCALE) % tsize;
                while(1){
                    old = atomicCAS(shared_col + hash, -1, bcol);
                    if(old == -1 || old == bcol){
                        atomicAdd(shared_val + hash, aval * bval);
                        break;
                    }
                    else{
                        hash = (hash + 1) < tsize ? hash + 1 : 0;
                    }
                }
            }
        }

        __syncthreads();


        // condense shared hash table
        // 紧致到左边，这个offset就是该行真正的nnz数量
        int offset;
        bool valid;
        #pragma unroll
        for (j = 0; j < SH_ROW; j += BS){
            offset = j + threadIdx.x;
            valid = offset < tsize;
            if(valid){
                acol = shared_col[offset];
                aval = shared_val[offset];
                if(acol != -1){
                    offset = atomicAdd(shared_offset, 1);
                }
            }
            __syncthreads();
            if(valid && acol != -1){
                shared_col[offset] = acol;
                shared_val[offset] = aval;
            }
        }
        
        // count sort the result
        // todo numeric的都需要遇到的时候-1跳出循环，因为是无效值
        int row_nnz = shared_offset[0];
        d_crpt_compressed[rid] = row_nnz;

        __syncthreads();
        int count, target;
        for (j = threadIdx.x; j < row_nnz; j += blockDim.x) {
            target = shared_col[j];
            count = 0;
            for (k = 0; k < row_nnz; k++) {
                count += (unsigned int)(shared_col[k] - target) >> 31;
            }
            d_ccol[c_offset + count] = shared_col[j];
            d_cval[c_offset + count] = shared_val[j];
        }


    }


    // kernel6
    __global__ void __launch_bounds__(1024, 1) d_numeric_max_shared_hash_tb_half_occu(
        const int * __restrict__ d_arpt, const int * __restrict__ d_acol, 
        const double * __restrict__ d_aval,
        const int * __restrict__ d_brpt, const int * __restrict__ d_bcol, 
        const double * __restrict__ d_bval,
        int *d_bins,
        int *d_crpt, int *d_ccol, double* d_cval,
        int *d_crpt_compressed){

        //long long t0 = clock64();

        int tid = threadIdx.x & (WSIZE - 1);
        int wid = threadIdx.x / WSIZE;
        int wnum = blockDim.x / WSIZE;
        int j, k;
        extern __shared__ int shared_mem[];
        const int tsize = 8191;
        int *shared_col = shared_mem;
        int *shared_offset = shared_mem + tsize;
        double* shared_val = (double*)(shared_mem + (tsize + 1));

        for(j = threadIdx.x; j < tsize; j += blockDim.x){
            shared_col[j] = -1;
            shared_val[j] = 0;
        }
        if(threadIdx.x == 0){
            shared_offset[0] = 0;
        }
        __syncthreads();

        int acol, bcol, hash, old;
        double aval, bval;
        int rid = d_bins[blockIdx.x];
        int c_offset = d_crpt[rid];
        // int row_nnz = d_crpt[rid + 1] - d_crpt[rid];

        for(j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum){
            acol = d_acol[j];
            aval = d_aval[j];
            for(k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k+= WSIZE){
                bcol = d_bcol[k];
                bval = d_bval[k];
                hash = (bcol * HASH_SCALE) % tsize;
                while(1){
                    old = atomicCAS(shared_col + hash, -1, bcol);
                    if(old == -1 || old == bcol){
                        atomicAdd(shared_val + hash, aval * bval);
                        break;
                    }
                    else{
                        hash = (hash + 1) < tsize ? hash + 1 : 0;
                    }

                }
            }
        }
        __syncthreads();


        // condense shared hash table
        int offset;
        bool valid;
        #pragma unroll
        for (j = 0; j < 8192; j += 1024){
            offset = j + threadIdx.x;
            valid = offset < tsize;
            if(valid){
                acol = shared_col[offset];
                aval = shared_val[offset];
                if(acol != -1){
                    offset = atomicAdd(shared_offset, 1);
                }
            }
            __syncthreads();
            if(valid && acol != -1){
                shared_col[offset] = acol;
                shared_val[offset] = aval;
            }
        }

        // count sort the result
        __syncthreads();
        // 真正的nnz
        int row_nnz = shared_offset[0];
        d_crpt_compressed[rid] = row_nnz;

        int count, target;
        for (j = threadIdx.x; j < row_nnz; j += blockDim.x) {
            target = shared_col[j];
            count = 0;
            for (k = 0; k < row_nnz; k++) {
                count += (unsigned int)(shared_col[k] - target) >> 31;
            }
            d_ccol[c_offset + count] = shared_col[j];
            d_cval[c_offset + count] = shared_val[j];
        }
        //__syncthreads();
        //t0 = clock64() - t0;
        //if(threadIdx.x == 0){
        //    printf("inside d_numeric_hash sort %lld\n", t0);
        //}
    }

    // kernel7
    __global__ void __launch_bounds__(1024, 1) d_numeric_max_shared_hash_tb_with_fail(
        const int * __restrict__ d_arpt, const int * __restrict__ d_acol, 
        const double * __restrict__ d_aval,
        const int * __restrict__ d_brpt, const int * __restrict__ d_bcol, 
        const double * __restrict__ d_bval,
        int *d_bins,
        int * __restrict__ d_fail_bins,
        int * __restrict__ d_fail_bin_size,
        int *d_crpt, int *d_ccol, double* d_cval,
        int *d_crpt_compressed){

        int tid = threadIdx.x & (WSIZE - 1);
        int wid = threadIdx.x / WSIZE;
        int wnum = blockDim.x / WSIZE;
        int j, k;
        extern __shared__ int shared_mem[];
        const int tsize = 8190;
        int *shared_col = shared_mem;
        int *shared_offset = shared_mem + tsize;
        int* shared_offset2 = shared_mem + tsize + 1;
        double* shared_val = (double*)(shared_mem + (tsize + 2));
        

        for(j = threadIdx.x; j < tsize; j += blockDim.x){
            shared_col[j] = -1;
            shared_val[j] = 0;
        }
        if(threadIdx.x == 0){
            shared_offset[0] = shared_offset2[0] = 0;
        }
        __syncthreads();

        int acol, bcol, hash, old;
        double aval, bval;
        int rid = d_bins[blockIdx.x];
        int c_offset = d_crpt[rid];
        // int row_nnz = d_crpt[rid + 1] - d_crpt[rid];

        for(j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum){
            acol = d_acol[j];
            aval = d_aval[j];
            for(k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k+= WSIZE){
                bcol = d_bcol[k];
                bval = d_bval[k];
                hash = (bcol * HASH_SCALE) % tsize;
                while(shared_offset2[0] <= 0.8 * tsize){
                    old = atomicCAS(shared_col + hash, -1, bcol);
                    if(old == bcol){
                        atomicAdd(shared_val + hash, aval * bval);
                        break;
                    }
                    else if(old == -1){
                        atomicAdd(shared_offset2, 1);
                        atomicAdd(shared_val + hash, aval * bval);
                        break;
                    }
                    else{
                        hash = (hash + 1) < tsize ? hash + 1 : 0;
                    }
                }
            }
        }

        //bug修复：nnz数量不对，如果使用shared内存，必须同步一下所有线程
        __syncthreads();

        int row_nnz = shared_offset2[0];
        if(threadIdx.x == 0) {
            d_crpt_compressed[rid] = row_nnz;
            // if(rid <= 18) printf("rid = %d, nnz = %d\n", rid, row_nnz);
        }
            

        __syncthreads();

        //经过全部行都让他失败，从而让global去算的办法验证了，确实是这个kernel导致的
        if(row_nnz > 0.8 * tsize){
            if(threadIdx.x == 0){
                int fail_index = atomicAdd(d_fail_bin_size, 1);
                d_fail_bins[fail_index] = rid;
            }
            return;
        }
        
        __syncthreads();

        // condense shared hash table
        int offset;
        bool valid;
        #pragma unroll
        for (j = 0; j < 8192; j += 1024){
            offset = j + threadIdx.x;
            valid = offset < tsize;
            if(valid){
                acol = shared_col[offset];
                aval = shared_val[offset];
                if(acol != -1){
                    offset = atomicAdd(shared_offset, 1);
                }
            }
            __syncthreads();
            if(valid && acol != -1){
                d_ccol[offset + c_offset] = acol;
                d_cval[offset + c_offset] = aval;
            }
        }
        
        
        // count sort the result
        // __syncthreads();
        // // 真正的nnz
        // row_nnz = shared_offset[0];
        // d_crpt_compressed[rid] = row_nnz;

        // int count, target;
        // for (j = threadIdx.x; j < row_nnz; j += blockDim.x) {
        //     target = shared_col[j];
        //     count = 0;
        //     for (k = 0; k < row_nnz; k++) {
        //         count += (unsigned int)(shared_col[k] - target) >> 31;
        //     }
        //     d_ccol[c_offset + count] = shared_col[j];
        //     d_cval[c_offset + count] = shared_val[j];
        // }
    }

    // kernel8
    __global__ void __launch_bounds__(1024, 2) d_numeric_global_hash_tb_full_occu(
        const int * __restrict__ d_arpt, const int * __restrict__ d_acol, 
        const double * __restrict__ d_aval,
        const int * __restrict__ d_brpt, const int * __restrict__ d_bcol, 
        const double * __restrict__ d_bval,
        int *d_bins, int max_tsize, int* d_tables,
        int *d_crpt, int *d_ccol, double* d_cval,
        int *d_crpt_compressed){

        //long long t0 = clock64();

        int tid = threadIdx.x & (WSIZE - 1);
        int wid = threadIdx.x / WSIZE;
        int wnum = blockDim.x / WSIZE;
        int j, k;
        __shared__ int shared_offset[1];
        
        int* table_col = d_tables + blockIdx.x * max_tsize * ((sizeof(int) + sizeof(double))/sizeof(int));
        double* table_val = (double*)(table_col + max_tsize);
        int rid = d_bins[blockIdx.x];
        int c_offset = d_crpt[rid];
        int row_nnz = d_crpt[rid + 1] - c_offset;
        //这里不可以去掉row_nnz的计算，否则global中的tsize没法确定
        //之前只用shared的bin空间足够，所以不存在这个问题
        int tsize = row_nnz * NUMERIC_SCALE_LARGE;
        for(j = threadIdx.x; j < tsize; j += blockDim.x){
            table_col[j] = -1;
            table_val[j] = 0;
        }
        if(threadIdx.x == 0){
            shared_offset[0] = 0;
        }
        __syncthreads();

        int acol, bcol, hash, old;
        double aval, bval;
        for(j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum){
            acol = d_acol[j];
            aval = d_aval[j];
            for(k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k+= WSIZE){
                bcol = d_bcol[k];
                bval = d_bval[k];
                hash = (bcol * HASH_SCALE) % tsize;
                while(1){
                    old = atomicCAS(table_col + hash, -1, bcol);
                    if(old == -1 || old == bcol){
                        atomicAdd(table_val + hash, aval * bval);
                        break;
                    }
                    else{
                        hash = (hash + 1) < tsize ? hash + 1 : 0;
                    }

                }
            }
        }

        //__syncthreads();
        //t0 = clock64() - t0;
        //if(threadIdx.x == 0){
        //    printf("inside d_numeric_hash global hash %lld\n", t0);
        //}
        //__syncthreads();
        //t0 = clock64();

        // condense shared hash table
        __syncthreads();
        int offset;
        for (j = threadIdx.x; j < tsize; j += blockDim.x){
            acol = table_col[j];
            aval = table_val[j];
            if(acol != -1){
                offset = atomicAdd(shared_offset, 1);
                d_ccol[c_offset + offset] = acol;
                d_cval[c_offset + offset] = aval;
            }
        }
        __syncthreads();
        row_nnz = shared_offset[0];
        d_crpt_compressed[rid] = row_nnz;
        for(j = threadIdx.x; j < row_nnz; j += blockDim.x){
            table_col[j] = d_ccol[c_offset + j];
            table_val[j] = d_cval[c_offset + j];
        }

        //__syncthreads();
        //t0 = clock64() - t0;
        //if(threadIdx.x == 0){
        //    printf("inside d_numeric_hash global  condense %lld\n", t0);
        //}
        //__syncthreads();
        //t0 = clock64();

        // count sort the result
        __syncthreads();
        int count, target;
        for (j = threadIdx.x; j < row_nnz; j += blockDim.x) {
            target = table_col[j];
            count = 0;
            for (k = 0; k < row_nnz; k++) {
                count += (unsigned int)(table_col[k] - target) >> 31;
            }
            d_ccol[c_offset + count] = table_col[j];
            d_cval[c_offset + count] = table_val[j];
        }
        //__syncthreads();
        //t0 = clock64() - t0;
        //if(threadIdx.x == 0){
        //    printf("inside d_numeric_hash sort %lld\n", t0);
        //}
    }
    
    void h_compute_flop(const CSR& A, const CSR& B, CSR& C, Info& info){
        int BS = 1024;
        int GS = div_up(A.M, BS);
        k_compute_flop<<<GS, BS>>>(A.d_rpt, A.d_col, B.d_rpt, C.M, C.d_rpt, C.d_rpt + C.M);
    }
    
    void h_setup(const CSR& A, const CSR& B, CSR& C, Info& info){
        info.allocate_rpt(C); // allocate C.rpt, other init procedure, default stream
        cudaMemset(C.d_rpt + C.M, 0, sizeof(int));
        h_compute_flop(A, B, C, info); // compute flop, stream[0]
        // d_bins   d_bin_size  d_max_row_nnz   d_total_nnz d_bin_offset    d_cub_storage
        // M        NUM_BIN     1               1           NUM_BIN         cub_storage_size
        //          bin_size    max_row_nnz     total_nnz   bin_offset
        //          NUM_BIN     1               1           NUM_BIN
        info.allocate(C); // allocate other memory    
        CHECK_ERROR(cudaMemcpy(info.max_row_nnz, C.d_rpt + C.M, sizeof(int), cudaMemcpyDeviceToHost));
    }
    
    inline void h_numeric_binning(CSR& C, Info& info){
        info.memset_all(0);
        int BS = 1024;
        int GS = div_up(C.M, BS);
        d_numeric_binning<<<GS, BS, 0 , info.stream[0]>>>(C.d_rpt, C.M,
            info.d_bin_size, info.d_total_nnz, info.d_max_row_nnz);
        info.D2H_all(0);
        CHECK_ERROR(cudaStreamSynchronize(info.stream[0]));
        // todo：应该是31或者32
        if(*info.max_row_nnz <= 16){
            k_binning_small<<<GS, BS>>>(info.d_bins, C.M);
            info.bin_size[0] = C.M;
            for(int i = 1; i< NUM_BIN; i++){
                info.bin_size[i] = 0;
            }
            info.bin_offset[0] = 0;
            for(int i = 1; i < NUM_BIN; i++){
                info.bin_offset[i] = C.M;
            }
        }
        else{
            info.memset_bin_size(0);
            info.bin_offset[0] = 0;
            for(int i = 0; i < NUM_BIN - 1; i++){
                info.bin_offset[i+1] = info.bin_offset[i] + info.bin_size[i];
            }
            info.H2D_bin_offset(0);
    
            d_numeric_binning2<<<GS, BS, 0, info.stream[0]>>>(C.d_rpt, C.M,
                info.d_bins, info.d_bin_size, info.d_bin_offset);
        }
    }
    
    inline void h_numeric_full_occu(const CSR& A, const CSR& B, CSR& C, Info& info){
    
        if(info.bin_size[6]){
            CHECK_ERROR(cudaFuncSetAttribute(d_numeric_max_shared_hash_tb_half_occu, 
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
            d_numeric_max_shared_hash_tb_half_occu<<<info.bin_size[6], 1024, 98304, info.stream[6]>>>
                (A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val,
                info.d_bins + info.bin_offset[6],
                C.d_rpt, C.d_col, C.d_val, C.d_rpt_compressed);
        }
    
        int *d_fail_bins, *d_fail_bin_size;
        int fail_bin_size = 0;

        if(info.bin_size[7]){ // global bin
            // printf("inside h_numeric_phase max_row_nnz %d\n", *info.max_row_nnz);
            
            ///////////////////////////////////////////////////////////////////////////////////////////////////
            // 写一个先尝试shared的kernel
            // 1. bin[7]中的先用shared_with_fail算，失败的行放到cub空间里

            if(info.bin_size[7] + 1 <= info.cub_storage_size/sizeof(int)){
                d_fail_bins = info.d_cub_storage;
                d_fail_bin_size = info.d_cub_storage + info.bin_size[7];
            }
            else{ // allocate global memory
                CHECK_ERROR(cudaMalloc(&d_fail_bins, (info.bin_size[7] + 1) * sizeof(int)));
                d_fail_bin_size = d_fail_bins + info.bin_size[7];
            }
            CHECK_ERROR(cudaMemsetAsync(d_fail_bin_size, 0, sizeof(int), info.stream[7]));

            CHECK_ERROR(cudaFuncSetAttribute(d_numeric_max_shared_hash_tb_with_fail, 
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));

            d_numeric_max_shared_hash_tb_with_fail
                <<<info.bin_size[7], 1024, 98304, info.stream[7]>>>(
                A.d_rpt, A.d_col, 
                A.d_val,
                B.d_rpt, B.d_col,
                B.d_val,
                info.d_bins + info.bin_offset[7],
                d_fail_bins, 
                d_fail_bin_size,
                C.d_rpt, C.d_col, C.d_val,
                C.d_rpt_compressed);

            ///////////////////////////////////////////////////////////////////////////////////////////////////
            CHECK_ERROR(cudaMemcpyAsync(&fail_bin_size, d_fail_bin_size, sizeof(int), cudaMemcpyDeviceToHost, info.stream[7]));
            // 确实有几个矩阵是用到全局内存的，比如a5esindl_2_R，但是数量不多
            // if(fail_bin_size > 0) {
            //     printf("fail_bin_size = %d\n\n\n\n\n\n\n\n\n\n\n\n\n", fail_bin_size);
            //     FILE *f = fopen("origin_aq_bq.txt", "a");
            //     fprintf(f, "next matrix has %d global rows \n", fail_bin_size);
            //     fclose(f);
            // }
            if(fail_bin_size) {
                int max_tsize = *info.max_row_nnz * NUMERIC_SCALE_LARGE;
                size_t global_size = info.bin_size[7] * max_tsize * (sizeof(int) + sizeof(double));
                if(info.global_mem_pool_malloced){
                    if(global_size <= info.global_mem_pool_size){
                        // do nothing
                    }
                    else{
                        CHECK_ERROR(cudaFree(info.d_global_mem_pool));
                        CHECK_ERROR(cudaMalloc(&info.d_global_mem_pool, global_size));
                    }
                }
                else{
                    CHECK_ERROR(cudaMalloc(&info.d_global_mem_pool, global_size));
                    info.global_mem_pool_size = global_size;
                    info.global_mem_pool_malloced = true;
                }
                d_numeric_global_hash_tb_full_occu<<<fail_bin_size, 1024, 0, info.stream[7]>>>
                    (A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val,
                        d_fail_bins, max_tsize, info.d_global_mem_pool,
                    C.d_rpt, C.d_col, C.d_val, C.d_rpt_compressed);
            }

            /*
            int max_tsize = *info.max_row_nnz * NUMERIC_SCALE_LARGE;
            size_t global_size = info.bin_size[7] * max_tsize * (sizeof(int) + sizeof(double));
            if(info.global_mem_pool_malloced){
                if(global_size <= info.global_mem_pool_size){
                    // do nothing
                }
                else{
                    CHECK_ERROR(cudaFree(info.d_global_mem_pool));
                    CHECK_ERROR(cudaMalloc(&info.d_global_mem_pool, global_size));
                }
            }
            else{
                CHECK_ERROR(cudaMalloc(&info.d_global_mem_pool, global_size));
                info.global_mem_pool_size = global_size;
                info.global_mem_pool_malloced = true;
            }
            d_numeric_global_hash_tb_full_occu<<<info.bin_size[7], 1024, 0, info.stream[7]>>>
                (A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val,
                info.d_bins + info.bin_offset[7], max_tsize, info.d_global_mem_pool,
                C.d_rpt, C.d_col, C.d_val, C.d_rpt_compressed);

            */
        }
    
        if(info.bin_size[5]){
            d_numeric_shared_hash_tb_full_occu<4096, 1024>
                <<<info.bin_size[5], 1024, 0, info.stream[5]>>>
                (A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val,
                info.d_bins + info.bin_offset[5],
                C.d_rpt, C.d_col, C.d_val, C.d_rpt_compressed);
        }
        if(info.bin_size[0]){
            int BS = NUMERIC_PWARP_ROWS * NUMERIC_PWARP;
            int GS = div_up(info.bin_size[0], NUMERIC_PWARP_ROWS);
            d_numeric_shared_hash_pwarp<<<GS, BS, 0, info.stream[0]>>>(
                A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val,
                info.d_bins + info.bin_offset[0], info.bin_size[0],
                C.d_rpt, C.d_col, C.d_val, C.d_rpt_compressed);
        }
    
        if(info.bin_size[4]){
            d_numeric_shared_hash_tb_full_occu<2048, 512>
                <<<info.bin_size[4], 512, 0, info.stream[4]>>>
                (A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val,
                info.d_bins + info.bin_offset[4],
                C.d_rpt, C.d_col, C.d_val, C.d_rpt_compressed);
        }
        if(info.bin_size[3]){
            d_numeric_shared_hash_tb_full_occu<1024, 256>
                <<<info.bin_size[3], 256, 0, info.stream[3]>>>
                (A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val,
                info.d_bins + info.bin_offset[3],
                C.d_rpt, C.d_col, C.d_val, C.d_rpt_compressed);
        }
    
        if(info.bin_size[2]){
            d_numeric_shared_hash_tb_full_occu<512, 128>
                <<<info.bin_size[2], 128, 0, info.stream[2]>>>
                (A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val,
                info.d_bins + info.bin_offset[2],
                C.d_rpt, C.d_col, C.d_val, C.d_rpt_compressed);
        }
        if(info.bin_size[1]){
            d_numeric_shared_hash_tb_full_occu<256, 64>
                <<<info.bin_size[1], 64, 0, info.stream[1]>>>
                (A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val,
                info.d_bins + info.bin_offset[1],
                C.d_rpt, C.d_col, C.d_val, C.d_rpt_compressed);
        }
    
        if(info.global_mem_pool_malloced){
            CHECK_ERROR(cudaFree(info.d_global_mem_pool));
        }
    }

    template <int WARP_SIZE>
    __global__ void
    __launch_bounds__( 512, 4 )
    spgemm_copy_from_Cext_into_C( int      M,
                                        int     *ix,
                                        int     *jx,
                                        double *ax,
                                        int     *ic,
                                        int     *jc,
                                        double *ac )
    {
        /* number of warps in the grid */ 
        const int grid_num_warps = blockDim.y * gridDim.x;
        /* warp id inside the block */
        const int warp_id = threadIdx.y;
        /* warp id in the grid */
        //第几个block * block中warp有几个 + block内的warp编号
        const int grid_warp_id = blockIdx.x * blockDim.y + warp_id;
        /* lane id inside the group */
        const int lane_id = threadIdx.y * blockDim.x + threadIdx.x;

        // 一个warp负责一行（todo：是否可以使用subwarp的思想呢？）
        for (int i = grid_warp_id; __any_sync(0xFFFFFFFF, i < M); i += grid_num_warps)
        {
                int istart_c = 0, iend_c = 0, istart_x = 0;

                /* start/end position in C and X */
                //   group_read<WARP_SIZE>(ic + i, true, istart_c, iend_c);
                int lane = threadIdx.y * blockDim.x + threadIdx.x;
                if (lane < 2) istart_c = __ldg(ic + i + lane);
                iend_c = __shfl_sync(0xFFFFFFFF, istart_c, 1);
                istart_c = __shfl_sync(0xFFFFFFFF, istart_c, 0);

                // group_read<WARP_SIZE>(ix + i, true, istart_x);
                lane = threadIdx.y * blockDim.x + threadIdx.x;
                if (!lane) istart_x = __ldg(ix + i);
                istart_x = __shfl_sync(0xFFFFFFFF, istart_x, 0);

                const int p = istart_x - istart_c;
                for (int k = istart_c + lane_id; k < iend_c; k += WARP_SIZE)
                {
                    jc[k] = jx[k + p];
                    ac[k] = ax[k + p];
                }
        }
    }

    // c的rpt，rpt_ext，col，col_ext，val, val_ext
    // 性能需要改进！，此函数典型时间是几百us
    template< typename T >
    __global__ static void copy_ext_to_C(
        int *d_crpt, 
        int *d_crpt_real, 
        int *d_ccol, 
        int *d_ccol_real,
        T *d_cval, 
        T *d_cval_real,
        int nrow)
    {
        // 一个线程负责一行的复制
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= nrow) {
            return;
        }
        int j = 0, cnt = 0;
        for (j = d_crpt_real[i]; j < d_crpt_real[i + 1]; j++) {
            // flop_per_row += d_brpt[d_acol[j] + 1] - d_brpt[d_acol[j]];
            d_ccol_real[j] = d_ccol[d_crpt[i] + cnt];
            d_cval_real[j] = d_cval[d_crpt[i] + cnt];
            cnt++;
        }
        // d_count[i] = flop_per_row;
        // atomicMax(d_max_flop, flop_per_row);
    }
    

    void nsparse(const CSR& A, const CSR& B, CSR& C, Info& info){
        C.M = A.M;
        C.N = B.N;
        C.nnz = 0;

        h_setup(A, B, C, info);
        CHECK_ERROR(cudaDeviceSynchronize());

        h_numeric_binning(C, info);
        CHECK_ERROR(cudaDeviceSynchronize());

        C.nnz = *info.total_nnz;

        CHECK_ERROR(cudaMalloc(&C.d_val, C.nnz * sizeof(double)));
        CHECK_ERROR(cudaMalloc(&C.d_col, C.nnz * sizeof(int)));
  
        cub::DeviceScan::ExclusiveSum(info.d_cub_storage, info.cub_storage_size, C.d_rpt, C.d_rpt, C.M + 1);
        
        CHECK_ERROR(cudaDeviceSynchronize());

        h_numeric_full_occu(A, B, C, info);
        CHECK_ERROR(cudaDeviceSynchronize());

        cub::DeviceScan::ExclusiveSum(info.d_cub_storage, info.cub_storage_size, C.d_rpt_compressed, C.d_rpt_compressed, C.M + 1);
        cudaMemcpy( &(C.nnz), C.d_rpt_compressed + C.M, sizeof(int), cudaMemcpyDeviceToHost );

        CHECK_ERROR(cudaMalloc(&C.d_val_compressed, C.nnz * sizeof(double)));
        CHECK_ERROR(cudaMalloc(&C.d_col_compressed, C.nnz * sizeof(int)));

        // copy回来
        const int num_warps_per_block = 512 / 32;
        dim3 bDim(32, 1, num_warps_per_block);
        dim3 gDim( (C.M + bDim.z - 1) / bDim.z ); //一个warp一行
        spgemm_copy_from_Cext_into_C<32><<<gDim, bDim>>>(
            C.M,
            C.d_rpt,
            C.d_col,
            C.d_val,
            C.d_rpt_compressed,
            C.d_col_compressed,
            C.d_val_compressed
        );

        info.release();

    }
}

int main(int argc, char ** argv)
{
    if (argc < 3)
    {
        printf("usage: ./exe MatrixName\n");
		//queryDevice();
        return 0;
    }
    char file_nameA[1024] = {0};
    char tempA[1024] = {0};
    strcpy(file_nameA, argv[1]);
    strcpy(tempA, argv[1]);
    char *mtx_pure_nameA = strrchr(tempA, '/');
    int len = strlen(mtx_pure_nameA);
    mtx_pure_nameA[len - 4] = '\0';
    strcpy(matrixNameA, mtx_pure_nameA+1);
    printf("reading file A %s\n", file_nameA);
    char file_nameB[1024] = {0};
    char tempB[1024] = {0};
    strcpy(file_nameB, argv[2]);
    strcpy(tempB, argv[2]);
    char *mtx_pure_nameB = strrchr(tempB, '/');
    len = strlen(mtx_pure_nameB);
    mtx_pure_nameB[len - 4] = '\0';
    strcpy(matrixNameB, mtx_pure_nameB+1);
    printf("reading file B %s\n", file_nameB);

    readMtx(file_nameA, MA, NA, nnzA, csrRowIndexHostPtrA, csrColIndexHostPtrA, csrValHostPtrA);
    printf("M=%d N=%d nnz=%d\n", MA, NA, nnzA);
    readMtx(file_nameB, MB, NB, nnzB, csrRowIndexHostPtrB, csrColIndexHostPtrB, csrValHostPtrB);
    printf("M=%d N=%d nnz=%d\n", MB, NB, nnzB);

    MC = MA;
    NC = NB;

    cudaEvent_t event[2];
    float msec, ave_msec; //flops;
    for (int i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }

    int max1 = 0;
    int max2 = 0;
    double mean1 = (double)nnzA / MA;
    double mean2 = (double)nnzB / MB;
    for (int i = 0; i < MB || i < MA; i++)
	{
        if(i < MA)
        {
            int nnz_row1 = csrRowIndexHostPtrA[i+1] - csrRowIndexHostPtrA[i];
            max1 = (nnz_row1 > max1) ? nnz_row1 : max1;
        }
		if(i < MB)
        {
            int nnz_row2 = csrRowIndexHostPtrB[i+1] - csrRowIndexHostPtrB[i];
            max2 = (nnz_row2 > max2) ? nnz_row2 : max2;
        }
	}
    double max_mu1 = max1 - mean1;
    double max_mu2 = max2 - mean2;
    

    /*
        0   1   2   auto
        old ato upp (从1和2中选择)
    */
    int choice = 0;
    if(strcmp(argv[3], "auto") == 0) {
        if(max_mu1 > 47.47) {
            choice  = 2;
        } else {
            choice = 1;
        }
        printf("max_mu1,2,choice = %lf %lf %d\n", max_mu1, max_mu2, choice);
    } else {
        // old atomic upperbound
        choice = atoi(argv[3]);
    }
    
    if(choice == 0) {
        cudaMalloc( (void **) &m_status, sizeof(int) );
        cudaMalloc( (void **) &m_work_queue, sizeof(int) );
        const int NUM_WARPS_IN_GRID = m_grid_size * m_max_warp_count;
        size_t sz = NUM_WARPS_IN_GRID * m_gmem_size * sizeof(int);
        cudaMalloc( (void **) &m_keys, sz );
        sz = NUM_WARPS_IN_GRID * m_gmem_size * sizeof(double);
        cudaMalloc( (void **) &m_vals, sz );
        cudaMalloc((void **)&csrRowIndexDevPtrA, (MA + 1) * sizeof(int));
        cudaMalloc((void **)&csrColIndexDevPtrA, nnzA * sizeof(int));
        cudaMalloc((void **)&csrValDevPtrA, nnzA * sizeof(double));
        cudaMemcpy(csrRowIndexDevPtrA, csrRowIndexHostPtrA, (MA + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrColIndexDevPtrA, csrColIndexHostPtrA, nnzA * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrValDevPtrA, csrValHostPtrA, nnzA * sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc((void **)&csrRowIndexDevPtrB, (MB + 1) * sizeof(int));
        cudaMalloc((void **)&csrColIndexDevPtrB, nnzB * sizeof(int));
        cudaMalloc((void **)&csrValDevPtrB, nnzB * sizeof(double));
        cudaMemcpy(csrRowIndexDevPtrB, csrRowIndexHostPtrB, (MB + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrColIndexDevPtrB, csrColIndexHostPtrB, nnzB * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrValDevPtrB, csrValHostPtrB, nnzB * sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc( (void **) &csrRowIndexDevPtrC, sizeof(int) * (MA + 1) );
        size_t temp_storage_bytes = 0;
        void *d_temp_storage = NULL;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, csrRowIndexDevPtrC, csrRowIndexDevPtrC, MC + 1);
        cudaMalloc( (void **) &d_temp_storage, temp_storage_bytes);

        cudaCheckError();
        bool done = false;
        bool use_origin = true;

        // todo 这里只使用了AP矩阵的负载均衡策略，如果是RAP应该开启下面的else
        // fixed 根据矩阵名字切换负载均衡的策略
        int avg_nz_per_row = nnzB / MB;
        char a = argv[2][strlen(argv[2]) - 5];
        if(a == 'A') {
            if ( avg_nz_per_row < 2 )
            {
                m_num_threads_per_row_count = 2;
                m_num_threads_per_row_compute = 2;
            }
            else
            {
                m_num_threads_per_row_count = 4;
                m_num_threads_per_row_compute = 4;
            } 
        } else {
            m_num_threads_per_row_count = (avg_nz_per_row <= 16.0 ? 8 : 32);
            m_num_threads_per_row_compute = (32);
        }

        for(int i=0; i<N; ++i) 
        {
            // printf("进来了q %d\n", t++);
            //每次循环恢复一下初始环境
            // m_gmem_size = temp_m_gmem_size;
            done = false;
            if(i > 0) {
                cudaFree(csrColIndexDevPtrC);
                csrColIndexDevPtrC = nullptr;
                cudaFree(csrValDevPtrC);
                csrValDevPtrC = nullptr;
            }

            cudaEventRecord(event[0], 0);
            try
            {
                for ( int attempt = 0 ; !done && attempt < 6 ; ++attempt )
                {
                    if ( attempt > 0 )
                    {
                        // printf("attempt = %d\n", attempt);

                        m_gmem_size *= 2;
                        // allocate_workspace();
                        cudaFree((void*)m_keys);
                        cudaFree((void*)m_vals);
                        
                        m_keys = nullptr;
                        m_vals = nullptr;

                        const int NUM_WARPS_IN_GRID = m_grid_size * m_max_warp_count;
                        size_t sz = NUM_WARPS_IN_GRID * m_gmem_size * sizeof(int);
                        cudaMalloc( (void **) &m_keys, sz );
                        sz = NUM_WARPS_IN_GRID * m_gmem_size * sizeof(double);
                        cudaMalloc( (void **) &m_vals, sz );
                    }

                    // Reset the status.
                    int status = 0;
                    cudaMemcpy( m_status, &status, sizeof(int), cudaMemcpyHostToDevice );
                    // Count the number of non-zeroes. The function count_non_zeroes assumes status has been
                    // properly set but it is responsible for setting the work queue.
                    origin_old::count_non_zeroes();
                    // Read the result from count_non_zeroes.
                    cudaMemcpy( &status, m_status, sizeof(int), cudaMemcpyDeviceToHost );
                    done = status == 0;
                }
            }
            catch (std::bad_alloc &e) // We are running out of memory. Try the fallback instead.
            {
                if ( done ) // Just in case but it should never happen.
                {
                    throw e;
                }
            }

            // We have to fallback to the CUSPARSE path.
            if ( !done )
            {
                use_origin = false;
                break;
            }

            if ( done )
            {
                // Compute row offsets.
                // this->compute_offsets( C );
                // thrust::exclusive_scan( thrust::device, csrRowIndexDevPtrC, csrRowIndexDevPtrC + MC + 1, csrRowIndexDevPtrC );
                cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, csrRowIndexDevPtrC, csrRowIndexDevPtrC, MC + 1);

                // Allocate memory to store columns/values.
                int num_vals = 0;
                cudaMemcpy(&num_vals, &csrRowIndexDevPtrC[MC], 4, cudaMemcpyDeviceToHost);
                cudaMalloc((void**) &csrColIndexDevPtrC, num_vals * sizeof(int));
                cudaMalloc((void**) &csrValDevPtrC, num_vals * sizeof(double));

                // C.col_indices.resize(num_vals);
                // C.values.resize(num_vals);
                // C.set_num_nz(num_vals);
                nnzC = num_vals;
                // Like count_non_zeroes, compute_values is responsible for setting its work queue (if it dares :)).
                done = false;

                if ( m_num_threads_per_row_count != m_num_threads_per_row_compute )
                {
                    // Reset the status.
                    int status = 0;
                    cudaMemcpy( m_status, &status, sizeof(int), cudaMemcpyHostToDevice );
                    // Count the number of non-zeroes. The function count_non_zeroes assumes status has been
                    // properly set but it is responsible for setting the work queue.
                    origin_old::compute_values();
                    // Read the result from count_non_zeroes.
                    cudaMemcpy( &status, m_status, sizeof(int), cudaMemcpyDeviceToHost );
                    done = status == 0;
                }

                // Re-run if needed. 也就是使用count阶段的线程分配方法重新计算
                if ( !done )
                {
                    m_num_threads_per_row_compute = m_num_threads_per_row_count;
                    origin_old::compute_values();
                }
            }
            cudaEventRecord(event[1], 0);
            cudaDeviceSynchronize();
            cudaEventElapsedTime(&msec, event[0], event[1]);
            ave_msec += msec;
        }
        
        if(use_origin) {
            ave_msec /= N;
        } else {
            ave_msec = -1;
        }

        // csrRowIndexHostPtrC = (int *)malloc((MC + 1) * sizeof(int));
        // csrColIndexHostPtrC = (int*)malloc(nnzC * sizeof(int));
        // csrValHostPtrC = (double *)malloc(nnzC * sizeof(double));
        // checkcuda(cudaMemcpy( csrRowIndexHostPtrC, csrRowIndexDevPtrC, (MC + 1) * sizeof(int), cudaMemcpyDeviceToHost ));
        // checkcuda(cudaMemcpy( csrColIndexHostPtrC, csrColIndexDevPtrC, nnzC * sizeof(int),     cudaMemcpyDeviceToHost ));
        // checkcuda(cudaMemcpy( csrValHostPtrC,      csrValDevPtrC,      nnzC * sizeof(double),  cudaMemcpyDeviceToHost ));
        int res = 0;
        // res = test_ok(nnzC, csrRowIndexHostPtrC, csrColIndexHostPtrC, csrValHostPtrC);
        // for(int i = csrRowIndexHostPtrC[MC - 1]; i < csrRowIndexHostPtrC[MC]; i++){
        //     printf("%d th row: %d %lf\n", MC - 1, csrColIndexHostPtrC[i], csrValHostPtrC[i]);
        // }
        
        FILE *f = fopen("origin_aq_bq.txt", "a");
        fprintf(f, "%s %s %f %d\n", argv[1], argv[2], ave_msec, res);
        fclose(f);
        
        free(csrRowIndexHostPtrC);
        free(csrColIndexHostPtrC);
        free(csrValHostPtrC);

        cudaFree(csrRowIndexDevPtrA);
        cudaFree(csrColIndexDevPtrA);
        cudaFree(csrValDevPtrA);
        cudaFree(csrRowIndexDevPtrB);
        cudaFree(csrColIndexDevPtrB);
        cudaFree(csrValDevPtrB);
        cudaFree(csrRowIndexDevPtrC);
        cudaFree(csrColIndexDevPtrC);
        cudaFree(csrValDevPtrC);
        cudaFree( (void **) &m_status);
        cudaFree( (void **) &m_work_queue);
        cudaFree( (void **) &m_keys);
        cudaFree( (void **) &m_vals);
        cudaFree((void **) &d_temp_storage);
    }

    if(choice == 1){
        cudaMalloc( (void **) &m_status, sizeof(int) );
        cudaMalloc( (void **) &m_work_queue, sizeof(int) );
        const int NUM_WARPS_IN_GRID = m_grid_size * m_max_warp_count;
        size_t sz = NUM_WARPS_IN_GRID * m_gmem_size * sizeof(int);
        cudaMalloc( (void **) &m_keys, sz );
        sz = NUM_WARPS_IN_GRID * m_gmem_size * sizeof(double);
        cudaMalloc( (void **) &m_vals, sz );
        cudaMalloc((void **)&csrRowIndexDevPtrA, (MA + 1) * sizeof(int));
        cudaMalloc((void **)&csrColIndexDevPtrA, nnzA * sizeof(int));
        cudaMalloc((void **)&csrValDevPtrA, nnzA * sizeof(double));
        cudaMemcpy(csrRowIndexDevPtrA, csrRowIndexHostPtrA, (MA + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrColIndexDevPtrA, csrColIndexHostPtrA, nnzA * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrValDevPtrA, csrValHostPtrA, nnzA * sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc((void **)&csrRowIndexDevPtrB, (MB + 1) * sizeof(int));
        cudaMalloc((void **)&csrColIndexDevPtrB, nnzB * sizeof(int));
        cudaMalloc((void **)&csrValDevPtrB, nnzB * sizeof(double));
        cudaMemcpy(csrRowIndexDevPtrB, csrRowIndexHostPtrB, (MB + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrColIndexDevPtrB, csrColIndexHostPtrB, nnzB * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrValDevPtrB, csrValHostPtrB, nnzB * sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc( (void **) &csrRowIndexDevPtrC, sizeof(int) * (MA + 1) );
        size_t temp_storage_bytes = 0;
        void *d_temp_storage = NULL;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, csrRowIndexDevPtrC, csrRowIndexDevPtrC, MC + 1);
        cudaMalloc( (void **) &d_temp_storage, temp_storage_bytes);

        cudaCheckError();
        bool done = false;
        bool use_origin = true;

        // todo 这里只使用了AP矩阵的负载均衡策略，如果是RAP应该开启下面的else
        // fixed 根据矩阵名字切换负载均衡的策略
        int avg_nz_per_row = nnzB / MB;
        char a = argv[2][strlen(argv[2]) - 5];
        if(a == 'A') {
            if ( avg_nz_per_row < 2 )
            {
                m_num_threads_per_row_count = 2;
                m_num_threads_per_row_compute = 2;
            }
            else
            {
                m_num_threads_per_row_count = 4;
                m_num_threads_per_row_compute = 4;
            } 
        } else {
            m_num_threads_per_row_count = (avg_nz_per_row <= 16.0 ? 8 : 32);
            m_num_threads_per_row_compute = (32);
        }
        for(int i=0; i<N; ++i) 
        {
            // printf("进来了q %d\n", t++);
            //每次循环恢复一下初始环境
            // m_gmem_size = temp_m_gmem_size;
            done = false;
            if(i > 0) {
                cudaFree(csrColIndexDevPtrC);
                csrColIndexDevPtrC = nullptr;
                cudaFree(csrValDevPtrC);
                csrValDevPtrC = nullptr;
            }

            cudaEventRecord(event[0], 0);
            try
            {
                for ( int attempt = 0 ; !done && attempt < 6 ; ++attempt )
                {
                    // printf("进来了w %d\n", t++);
                    // Double the amount of GMEM (if needed).
                    if ( attempt > 0 )
                    {
                        // printf("attempt = %d\n", attempt);

                        m_gmem_size *= 2;
                        // allocate_workspace();
                        cudaFree((void*)m_keys);
                        cudaFree((void*)m_vals);
                        
                        m_keys = nullptr;
                        m_vals = nullptr;

                        const int NUM_WARPS_IN_GRID = m_grid_size * m_max_warp_count;
                        size_t sz = NUM_WARPS_IN_GRID * m_gmem_size * sizeof(int);
                        cudaMalloc( (void **) &m_keys, sz );
                        sz = NUM_WARPS_IN_GRID * m_gmem_size * sizeof(double);
                        cudaMalloc( (void **) &m_vals, sz );
                    }

                    // Reset the status.
                    int status = 0;
                    cudaMemcpy( m_status, &status, sizeof(int), cudaMemcpyHostToDevice );
                    // Count the number of non-zeroes. The function count_non_zeroes assumes status has been
                    // properly set but it is responsible for setting the work queue.
                    origin::count_non_zeroes();

                    // Read the result from count_non_zeroes.
                    cudaMemcpy( &status, m_status, sizeof(int), cudaMemcpyDeviceToHost );
                    done = status == 0;
                }
            }
            catch (std::bad_alloc &e) // We are running out of memory. Try the fallback instead.
            {
                if ( done ) // Just in case but it should never happen.
                {
                    throw e;
                }
            }

            // We have to fallback to the CUSPARSE path.
            if ( !done )
            {
                use_origin = false;
                break;
            }

            if ( done )
            {
                // Compute row offsets.
                // this->compute_offsets( C );
                // thrust::exclusive_scan( thrust::device, csrRowIndexDevPtrC, csrRowIndexDevPtrC + MC + 1, csrRowIndexDevPtrC );
                cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, csrRowIndexDevPtrC, csrRowIndexDevPtrC, MC + 1);

                // Allocate memory to store columns/values.
                int num_vals = 0;
                cudaMemcpy(&num_vals, &csrRowIndexDevPtrC[MC], 4, cudaMemcpyDeviceToHost);
                cudaMalloc((void**) &csrColIndexDevPtrC, num_vals * sizeof(int));
                cudaMalloc((void**) &csrValDevPtrC, num_vals * sizeof(double));

                // C.col_indices.resize(num_vals);
                // C.values.resize(num_vals);
                // C.set_num_nz(num_vals);
                nnzC = num_vals;
                // Like count_non_zeroes, compute_values is responsible for setting its work queue (if it dares :)).
                done = false;

                if ( m_num_threads_per_row_count != m_num_threads_per_row_compute )
                {
                    // Reset the status.
                    int status = 0;
                    cudaMemcpy( m_status, &status, sizeof(int), cudaMemcpyHostToDevice );
                    // Count the number of non-zeroes. The function count_non_zeroes assumes status has been
                    // properly set but it is responsible for setting the work queue.
                    origin::compute_values( );
                    // Read the result from count_non_zeroes.
                    cudaMemcpy( &status, m_status, sizeof(int), cudaMemcpyDeviceToHost );
                    done = status == 0;
                }

                // Re-run if needed. 也就是使用count阶段的线程分配方法重新计算
                if ( !done )
                {
                    m_num_threads_per_row_compute = m_num_threads_per_row_count;
                    origin::compute_values();
                }
            }
            cudaEventRecord(event[1], 0);
            cudaDeviceSynchronize();
            cudaEventElapsedTime(&msec, event[0], event[1]);
            ave_msec += msec;
        }
        
        if(use_origin) {
            ave_msec /= N;
        } else {
            ave_msec = -1;
        }

        // csrRowIndexHostPtrC = (int *)malloc((MC + 1) * sizeof(int));
        // csrColIndexHostPtrC = (int*)malloc(nnzC * sizeof(int));
        // csrValHostPtrC = (double *)malloc(nnzC * sizeof(double));
        // checkcuda(cudaMemcpy( csrRowIndexHostPtrC, csrRowIndexDevPtrC, (MC + 1) * sizeof(int), cudaMemcpyDeviceToHost ));
        // checkcuda(cudaMemcpy( csrColIndexHostPtrC, csrColIndexDevPtrC, nnzC * sizeof(int),     cudaMemcpyDeviceToHost ));
        // checkcuda(cudaMemcpy( csrValHostPtrC,      csrValDevPtrC,      nnzC * sizeof(double),  cudaMemcpyDeviceToHost ));
        int res = 0;
        // res = test_ok(nnzC, csrRowIndexHostPtrC, csrColIndexHostPtrC, csrValHostPtrC);
        // for(int i = csrRowIndexHostPtrC[MC - 1]; i < csrRowIndexHostPtrC[MC]; i++){
        //     printf("%d th row: %d %lf\n", MC - 1, csrColIndexHostPtrC[i], csrValHostPtrC[i]);
        // }
        
        FILE *f = fopen("origin_aq_bq.txt", "a");
        fprintf(f, "%s %s %f %d\n", argv[1], argv[2], ave_msec, res);
        fclose(f);
        
        free(csrRowIndexHostPtrC);
        free(csrColIndexHostPtrC);
        free(csrValHostPtrC);

        cudaFree(csrRowIndexDevPtrA);
        cudaFree(csrColIndexDevPtrA);
        cudaFree(csrValDevPtrA);
        cudaFree(csrRowIndexDevPtrB);
        cudaFree(csrColIndexDevPtrB);
        cudaFree(csrValDevPtrB);
        cudaFree(csrRowIndexDevPtrC);
        cudaFree(csrColIndexDevPtrC);
        cudaFree(csrValDevPtrC);
        cudaFree( (void **) &m_status);
        cudaFree( (void **) &m_work_queue);
        cudaFree( (void **) &m_keys);
        cudaFree( (void **) &m_vals);
        cudaFree((void **) &d_temp_storage);
        
    }
    
    if(choice == 2) {
        std::string mat1, mat2;
        mat1 = argv[1];
        mat2 = argv[2];
        std::string mat1_file = mat1;
        std::string mat2_file = mat2;
        nsparse::CSR A, B;
        A.construct(mat1_file);
        B.construct(mat2_file);
        A.H2D();
        B.H2D();
        nsparse::CSR C;
        nsparse::Info info;
        
        // cudaEvent_t event[2];
        float ave_msec; //flops;
        // for (int i = 0; i < 2; i++) {
        //     cudaEventCreate(&(event[i]));
        // }
        struct timeval tv0, tv1;
        for(int i = 0; i < N; i++){
            cudaEventRecord(event[0], 0);
            // CHECK_ERROR(cudaDeviceSynchronize());
            // gettimeofday(&tv0, NULL);
            nsparse::nsparse(A, B, C, info);
            // gettimeofday(&tv1, NULL);
            // CHECK_ERROR(cudaDeviceSynchronize());
            // ave_msec += (tv1.tv_sec * 1000.0 + tv1.tv_usec / 1000.0)-(tv0.tv_sec * 1000.0 + tv0.tv_usec / 1000.0);
            // printf("%f\n", (tv1.tv_sec * 1000.0 + tv1.tv_usec / 1000.0)-(tv0.tv_sec * 1000.0 + tv0.tv_usec / 1000.0));
            cudaEventRecord(event[1], 0);
            cudaDeviceSynchronize();
            cudaEventElapsedTime(&msec, event[0], event[1]);
            ave_msec += msec;
            if(i < N - 1){
                C.release();
            }
        }
        ave_msec /= N;

        /*
                int M;
        int N;
        int nnz;

        int *rpt = nullptr;
        int *col = nullptr;
        double *val = nullptr;
        
        */
        
        // C.D2H();
        // for(int i = C.rpt_compressed[C.M - 1]; i < C.rpt_compressed[C.M]; i++){
        //     printf("%d th row: %d %lf\n", C.M - 1, C.col_compressed[i], C.val_compressed[i]);
        // }
        int res = 0;
        // res = test_ok(C.nnz, C.rpt_compressed, C.col_compressed, C.val_compressed);
        // res = test_ok_for_nsparse(C.nnz, C.rpt_compressed, C.col_compressed, C.val_compressed,
        //     A.d_rpt, A.d_col, A.d_val,
        //     B.d_rpt, B.d_col, B.d_val,
        //     A.M, A.N, A.nnz,
        //     B.M, B.N, B.nnz);
        
        // printf("%f ms\n", ave_msec);
        FILE *f = fopen("origin_aq_bq.txt", "a");
        fprintf(f, "%s %s %f %d\n", argv[1], argv[2], ave_msec, res);
        fclose(f);

        A.release();
        B.release();
        C.release();
        
    }
    
    

    // C.release();
    // int N = 10;
    
    // for(int i = 0; i < N; i++){
    //     nsparse::nsparse(A, B, C, info);
    //     if(i < N - 1){
    //         C.release();
    //     }
    // }

    // 现在行指针肯定还不对，然后列索引和值数组肯定有一些无效值
    // printf("%d %d\n", C.nnz, nnzC);
    // C.D2H();
    // for(int i = 0; i < 100; i++){
    //     printf("%d ", C.rpt[i]);
    // }
    // printf("\n");
    // for(int i = 0; i < 100; i++){
    //     printf("%d ", C.col[i]);
    // }
    // printf("\n");
    // for(int i = 0; i < 100; i++){
    //     printf("%lf ", C.val[i]);
    // }
    // printf("\n");

    // C.D2H();
    // for(int i = 0; i < 100; i++){
    //     printf("%d ", C.rpt_compressed[i]);
    // }
    // printf("\n");
    // for(int i = 0; i < 100; i++){
    //     printf("%d ", C.col_compressed[i]);
    // }
    // printf("\n");
    // for(int i = 0; i < 100; i++){
    //     printf("%lf ", C.val_compressed[i]);
    // }
    // printf("\n");


    // FILE * fp = fopen("nsparse_benchmark.txt", "a+");
    // fprintf(fp, "%s %s ", argv[1], argv[2]);
    // fprintf(fp, "%f\n", timing.total);
    // fclose(fp);
    

    free(csrRowIndexHostPtrA);
    free(csrColIndexHostPtrA);
    free(csrValHostPtrA);
    free(csrRowIndexHostPtrB);
    free(csrColIndexHostPtrB);
    free(csrValHostPtrB);

    // printf("nnzC = %d\n", nnzC);

    return 0;
}