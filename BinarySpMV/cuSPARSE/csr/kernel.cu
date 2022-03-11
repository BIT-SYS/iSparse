//Example: Application using C++ and the CUSPARSE library
//-------------------------------------------------------
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
#include "cusparse.h"
#include "mmio.h"
#include <omp.h>
#include <dirent.h>

using namespace std;

#define NUM_TRANSFER 50
#define NUM_RUN 100

inline void checkcuda(cudaError_t result)
{
	if (result != cudaSuccess) {
		printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		printf("hello");
	}
}

inline void checkcusparse(cusparseStatus_t result)
{
	if (result != CUSPARSE_STATUS_SUCCESS) {
		printf("CUSPARSE Error, error_code =  %d\n", result);
	}
}

double average(int n, double *data)
{
    double ave = 0.0;
    for(int i = 0; i < n; i++)
        ave += data[i];
    
    return ave / n;
}

double variance(int n, double ave, double *data)
{
    double var = 0.0;
    for (int i = 0; i < n; i++) {
        double temp = data[i] - ave;
        var += (temp * temp);
    }
    
    return var / n;
}

int readMtx(char *filename, int &m, int &n, int &nnzA, int *&csrRowIdxA, int *&csrColIdxA,
	float *&csrValA);
int cudaSpmv(int *&csrRowIndexHostPtr, int *&csrColIndexHostPtr,
			 float *&csrValHostPtr, float *&xHostPtr, float *&yHostPtr);
void queryDevice();

/*** Declaration ***/
int M, N, nnz;				 // M (row number), N (column number), nnz (Number of Non-Zero members)
char matrixName[1024] = {0}; 

// host variables
int *csrRowIndexHostPtr = 0; // coo format row index
int *csrColIndexHostPtr = 0; // coo format column index
float *csrValHostPtr = 0;	 // coo format value index
float *xHostPtr = 0;		 // the multiplied vector
float *yHostPtr = 0;		 // the result vector

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		printf("usage: ./exe MatrixFile\n");
		return 0;
	}

	printf("%s %s\n", argv[0], argv[1]);

	queryDevice();

	char matrix_dir[1024] = {0};
	strcpy(matrix_dir, argv[1]);

	//find matrix file
	DIR *matrix_dir_handle;
	struct dirent *matrix_entry;
	matrix_dir_handle = opendir(matrix_dir);

	int counter = 0, error_count = 0;
	while ((matrix_entry = readdir(matrix_dir_handle)) != NULL) {
		if (strcmp(matrix_entry->d_name, "..") != 0 && strcmp(matrix_entry->d_name, ".") != 0) {
			char source[1024] = {0};
			strcpy(source, argv[1]);
			strcat(source, "/");
			strcat(source, matrix_entry->d_name);

			strcpy(matrixName, matrix_entry->d_name);
			printf("%s\n", matrixName);

			//deal every matrix. source is the complete path name of the matrix
			readMtx(source, M, N, nnz, csrRowIndexHostPtr, csrColIndexHostPtr, csrValHostPtr);

			xHostPtr = (float *)malloc(N * sizeof(float));
			for (int i = 0; i < N; i++) xHostPtr[i] = 1.0;
			yHostPtr = (float *)malloc(M * sizeof(float));

			cudaSpmv(csrRowIndexHostPtr, csrColIndexHostPtr, csrValHostPtr, xHostPtr, yHostPtr);

			float * y_ref = (float *)malloc(sizeof(float) * M);
			for (int i = 0; i < M; i++)
			{
				float sum = 0.0;
				for (int j = csrRowIndexHostPtr[i]; j < csrRowIndexHostPtr[i+1]; j++)
					sum += xHostPtr[csrColIndexHostPtr[j]] * csrValHostPtr[j];
				y_ref[i] = sum;
			}

			for (int i = 0; i < M; i++)
				if ( abs(y_ref[i] - yHostPtr[i]) > 1e-6 ) {
					error_count++;
					break;
				}

			free(csrRowIndexHostPtr);
			free(csrColIndexHostPtr);
			free(csrValHostPtr);
			free(xHostPtr);
			free(yHostPtr);
			free(y_ref);
			counter++;
			printf("%d finished, %d error\n\n\n",counter, error_count);
		} //end if
	}	  //end while

	printf("\nTest finished!  %d matrices has been test, error_count = %d\n", counter, error_count);
	printf("--------------------------------------------------\n\n");

	return 0;
}

int readMtx(char *filename, int &m, int &n, int &nnzA, int *&csrRowPtrA, int *&csrColIdxA,
			 float *&csrValA)
{
	int ret_code;
	MM_typecode matcode;

	FILE *f = NULL;
	int nnzA_mtx_report;
	int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;
	// load matrix
	if ((f = fopen(filename, "r")) == NULL)
		return -1;

	if (mm_read_banner(f, &matcode) != 0) {
		printf("Could not process Matrix Market banner.\n");
		return -2;
	}

	if (mm_is_complex(matcode)) {
		printf("Sorry, data type 'COMPLEX' is not supported. \n");
		return -3;
	}

	if (mm_is_pattern(matcode)) {
		isPattern = 1; printf("type = Pattern.\n");
	}

	if (mm_is_real(matcode)) {
		isReal = 1; printf("type = real.\n");
	}

	if (mm_is_integer(matcode)) {
		isInteger = 1; printf("type = integer.\n");
	}

	ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report);
	if (ret_code != 0)
		return -4;

	if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode)) {
		isSymmetric = 1;
		printf("symmetric = true.\n");
	}
	else {
		printf("symmetric = false.\n");
	}

	int *csrRowPtrA_counter = (int *)malloc((m + 1) * sizeof(int));
	memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(int));

	int *csrRowIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
	int *csrColIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
	float *csrValA_tmp = (float *)malloc(nnzA_mtx_report * sizeof(float));

	for (int i = 0; i < nnzA_mtx_report; i++)
	{
		int idxi, idxj;
		double fval;
		int ival;

		if (isReal)
			fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
		else if (isInteger)
		{
			fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
			fval = ival;
		}
		else if (isPattern)
		{
			fscanf(f, "%d %d\n", &idxi, &idxj);
			fval = 1.0;
		}

		// adjust from 1-based to 0-based
		idxi--;
		idxj--;

		csrRowPtrA_counter[idxi]++;
		csrRowIdxA_tmp[i] = idxi;
		csrColIdxA_tmp[i] = idxj;
		csrValA_tmp[i] = fval;
	}

	if (f != stdin)
		fclose(f);

	if (isSymmetric)
	{
		for (int i = 0; i < nnzA_mtx_report; i++)
		{
			if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
				csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
		}
	}

	// exclusive scan for csrRowPtrA_counter
	int old_val, new_val;

	old_val = csrRowPtrA_counter[0];
	csrRowPtrA_counter[0] = 0;
	for (int i = 1; i <= m; i++)
	{
		new_val = csrRowPtrA_counter[i];
		csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i - 1];
		old_val = new_val;
	}

	nnzA = csrRowPtrA_counter[m];
	csrRowPtrA = (int *)malloc((m + 1) * sizeof(int));
	memcpy(csrRowPtrA, csrRowPtrA_counter, (m + 1) * sizeof(int));
	memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(int));

	csrColIdxA = (int *)malloc(nnzA * sizeof(int));
	csrValA = (float *)malloc(nnzA * sizeof(float));

	if (isSymmetric)
	{
		for (int i = 0; i < nnzA_mtx_report; i++)
		{
			if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
			{
				int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
				csrColIdxA[offset] = csrColIdxA_tmp[i];
				csrValA[offset] = csrValA_tmp[i];
				csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;

				offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
				csrColIdxA[offset] = csrRowIdxA_tmp[i];
				csrValA[offset] = csrValA_tmp[i];
				csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
			}
			else
			{
				int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
				csrColIdxA[offset] = csrColIdxA_tmp[i];
				csrValA[offset] = csrValA_tmp[i];
				csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
			}
		}
	}
	else
	{
		for (int i = 0; i < nnzA_mtx_report; i++)
		{
			int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
			csrColIdxA[offset] = csrColIdxA_tmp[i];
			csrValA[offset] = csrValA_tmp[i];
			csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
		}
	}

	// free tmp space
	free(csrColIdxA_tmp);
	free(csrValA_tmp);
	free(csrRowIdxA_tmp);
	free(csrRowPtrA_counter);
	
	return 0;
}

int cudaSpmv(int *&csrRowIndexHostPtr, int *&csrColIndexHostPtr, float *&csrValHostPtr, float *&xHostPtr, float *&yHostPtr)
{
	// GPU variables
	cusparseHandle_t handle = 0;  // cusparse handle, if you want to use cusparse ,you must create a cusparse handle
	cusparseMatDescr_t descr = 0; // a matrix descriptor used for multiplication
	float done = 1.0;			  //float number 1
	float dzero = 0.0;			  //float number 0
	int memory_size = 0;

	int *csrRowIndexGpuPtr = 0;	  // coo format row index
	int *csrColIndexGpuPtr = 0;	  // coo format column index
	float *csrValGpuPtr = 0;	  // coo format value index
	float *xGpuPtr = 0;			  // the multiplied vector
	float *yGpuPtr = 0;			  // the result vector

	/*** Allocate GPU Memory ***/
	checkcuda(cudaMalloc((void **)&csrRowIndexGpuPtr, (M+1) * sizeof(int)));
	checkcuda(cudaMalloc((void **)&csrColIndexGpuPtr, nnz * sizeof(int)));
	checkcuda(cudaMalloc((void **)&csrValGpuPtr, nnz * sizeof(float)));
	checkcuda(cudaMalloc((void **)&xGpuPtr, N * sizeof(float)));
	checkcuda(cudaMalloc((void **)&yGpuPtr, M * sizeof(float)));

	memory_size += nnz * 2;		  // memory of csrValGpuPtr, csrColIndexGpuPtr
	memory_size += M + 1; // memory of csrRowIndexGpuPtr
	printf("memory_size = %.4fMB.\n", (float)(memory_size)*4 / 1024 / 1024);

	double transfer_time = 0.0, calculate_time = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start); // create event
	cudaEventCreate(&stop);

	// initialize cusparse library
	checkcusparse(cusparseCreate(&handle));
	// initialize matrix descriptor
	checkcusparse(cusparseCreateMatDescr(&descr));
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

	// copy x vector from CPU to GPU
	checkcuda(cudaMemcpy(xGpuPtr, xHostPtr, (N * sizeof(float)), cudaMemcpyHostToDevice));

	double all_transfer_time[NUM_TRANSFER];
	for (int i = 0; i < NUM_TRANSFER; i++) {
        float temp = 0.0;
        cudaEventRecord(start, 0);
		// copy CSR-based matrix from CPU to GPU
		checkcuda(cudaMemcpy(csrRowIndexGpuPtr, csrRowIndexHostPtr, ((M + 1) * sizeof(int)), cudaMemcpyHostToDevice));
		checkcuda(cudaMemcpy(csrColIndexGpuPtr, csrColIndexHostPtr, (nnz * sizeof(int)), cudaMemcpyHostToDevice));
		checkcuda(cudaMemcpy(csrValGpuPtr, csrValHostPtr, (nnz * sizeof(float)), cudaMemcpyHostToDevice));
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp, start, stop);
        all_transfer_time[i] = temp;
	}
    transfer_time = average(NUM_TRANSFER, all_transfer_time);
    double var_transfer_time = variance(NUM_TRANSFER, transfer_time, all_transfer_time);
	printf("%s transmission time %.6f ", matrixName, transfer_time);

    /*** run one time for checking result ***/
    checkcuda(cudaMemset(yGpuPtr, 0.0, M * sizeof(float)));
    checkcusparse(cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, nnz, &done, descr, csrValGpuPtr, csrRowIndexGpuPtr, csrColIndexGpuPtr, xGpuPtr, &dzero, yGpuPtr));
    checkcuda(cudaMemcpy(yHostPtr, yGpuPtr, (M * sizeof(float)), cudaMemcpyDeviceToHost));
	
    double all_spmv_time[NUM_RUN];
	for (int i = 0; i < NUM_RUN; i++)
	{
        float temp = 0.0;
        cudaEventRecord(start, 0); 
		/*** CSR SpMV ***/
		checkcusparse(cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, nnz, &done, descr, csrValGpuPtr, csrRowIndexGpuPtr, csrColIndexGpuPtr, xGpuPtr, &dzero, yGpuPtr));
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp, start, stop);
        all_spmv_time[i] = temp;
    }
	calculate_time = average(NUM_RUN, all_spmv_time);
    double var_calculate_time = variance(NUM_RUN, calculate_time, all_spmv_time);
	printf("calculation time %.6f\n", calculate_time);

	// FILE *fresult = fopen("cusparse_csr_time_float_V100.txt", "a+");
	// if (fresult == NULL) {
	// 	printf("Create file failed.\n ");
	// }
	// else {
	// 	fprintf(fresult, "%s %.6f %.6f %.6f %.6f\n", matrixName, transfer_time, var_transfer_time, calculate_time, var_calculate_time);
	// 	fclose(fresult);
	// }

	/*** Release Resource ***/
	// destroy matrix descriptor
	checkcusparse(cusparseDestroyMatDescr(descr));
	descr = 0;
	// destroy handle
	checkcusparse(cusparseDestroy(handle));
	handle = 0;
	cudaFree(csrRowIndexGpuPtr);
	cudaFree(csrColIndexGpuPtr);
	cudaFree(csrValGpuPtr);
	cudaFree(xGpuPtr);
	cudaFree(yGpuPtr);
	return 0;
}

void queryDevice()
{
	cudaDeviceProp deviceProp;
	int deviceCount = 0;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&deviceCount);
	cout << "cudaError = " << cudaError << endl;
	for (int i = 0; i < deviceCount; i++)
	{
		cudaError = cudaGetDeviceProperties(&deviceProp, i);
		cout << "Device " << i << endl;
		cout << "Device name: " << deviceProp.name << endl;
		cout << "Total global memory (MB) : " << deviceProp.totalGlobalMem / 1024 / 1024 << endl;
		cout << "Share memory per block (KB) : " << deviceProp.sharedMemPerBlock / 1024 << endl;
		cout << "Number of registers per block (KB) : " << deviceProp.regsPerBlock << endl;
		cout << "Maximum threads per block : " << deviceProp.maxThreadsPerBlock << endl;
		cout << "Compute capability : " << deviceProp.major << "." << deviceProp.minor << endl;
		cout << "Number of multi-processor : " << deviceProp.multiProcessorCount << endl;
	}
	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		printf("cudaSetDevice failed!");

	int device = -1;
	cudaStatus = cudaGetDevice(&device);
	if (cudaStatus != cudaSuccess)
		printf("cudaGetDevice failed!");
	cout << "\nThe device now beening used is device " << device << endl;
}
