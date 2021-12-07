#ifndef __FORMAT__
#define __FORMAT__
#include <stdlib.h>
#include <stdio.h>
#include "mmio.h"
#include <iostream>
#include <string>
typedef unsigned int uint;
using namespace std;

template <typename ValueType>
class SpM
{
public:
	uint nrows;
	uint ncols;
	uint nnz;
	uint *rows;
	uint *cols;
	ValueType *vals;
	SpM() : nrows(0), ncols(0), nnz(0), rows(nullptr), cols(nullptr), vals(nullptr){};
	int readMtx(char *filename);
	void matrix_host2device(const SpM<ValueType> &A);
};

template <typename ValueType>
void SpM<ValueType>::matrix_host2device(const SpM<ValueType> &A)
{
	nrows = A.nrows;
	ncols = A.ncols;
	nnz = A.nnz;
	cudaMalloc((void **)&rows, (nrows + 1) * sizeof(uint));
	cudaMalloc((void **)&cols, nnz * sizeof(uint));
	cudaMalloc((void **)&vals, nnz * sizeof(ValueType));

	cudaMemcpy(rows, A.rows, (nrows + 1) * sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(cols, A.cols, nnz * sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(vals, A.vals, nnz * sizeof(ValueType), cudaMemcpyHostToDevice);
}

template <typename ValueType>
int SpM<ValueType>::readMtx(char *filename)
{
	int ret_code = 0;
	MM_typecode matcode;

	FILE *f = NULL;
	uint nnzA_mtx_report = 0;
	int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;
	// load matrix
	if ((f = fopen(filename, "r")) == NULL)
		return -1;

	if (mm_read_banner(f, &matcode) != 0)
	{
		printf("Could not process Matrix Market banner.\n");
		return -2;
	}

	if (mm_is_complex(matcode))
	{
		printf("Sorry, data type 'COMPLEX' is not supported. \n");
		return -3;
	}

	if (mm_is_pattern(matcode))
	{
		isPattern = 1;
		printf("type = Pattern.\n");
	}

	if (mm_is_real(matcode))
	{
		isReal = 1;
		printf("type = real.\n");
	}

	if (mm_is_integer(matcode))
	{
		isInteger = 1;
		printf("type = integer.\n");
	}

	ret_code = mm_read_mtx_crd_size(f, &nrows, &ncols, &nnzA_mtx_report);
	if (ret_code != 0)
		return -4;

	if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
	{
		isSymmetric = 1;
		printf("symmetric = true.\n");
	}
	else
	{
		printf("symmetric = false.\n");
	}

	uint *csrRowPtrA_counter = (uint *)malloc((nrows + 1) * sizeof(uint));
	memset(csrRowPtrA_counter, 0, (nrows + 1) * sizeof(uint));

	uint *csrRowIdxA_tmp = (uint *)malloc(nnzA_mtx_report * sizeof(uint));
	memset(csrRowIdxA_tmp, 0, nnzA_mtx_report * sizeof(uint));
	uint *csrColIdxA_tmp = (uint *)malloc(nnzA_mtx_report * sizeof(uint));
	memset(csrColIdxA_tmp, 0, nnzA_mtx_report * sizeof(int));
	ValueType *csrValA_tmp = (ValueType *)malloc(nnzA_mtx_report * sizeof(ValueType));
	memset(csrValA_tmp, 0.0, nnzA_mtx_report * sizeof(ValueType));

	for (uint i = 0; i < nnzA_mtx_report; i++)
	{
		uint idxi = 0, idxj = 0;
		ValueType fval = 0.0;
		int ival = 0;

		if (isReal)
			fscanf(f, "%u %u %lg\n", &idxi, &idxj, &fval);
		else if (isInteger)
		{
			fscanf(f, "%u %u %d\n", &idxi, &idxj, &ival);
			fval = ival;
		}
		else if (isPattern)
		{
			fscanf(f, "%u %u\n", &idxi, &idxj);
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
		for (uint i = 0; i < nnzA_mtx_report; i++)
		{
			if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
				csrRowPtrA_counter[csrColIdxA_tmp[i]]++; //对称后，j变成i
		}
	}

	// exclusive scan for csrRowPtrA_counter
	uint old_val = 0, new_val = 0;

	old_val = csrRowPtrA_counter[0];
	csrRowPtrA_counter[0] = 0;
	for (uint i = 1; i <= nrows; i++)
	{
		new_val = csrRowPtrA_counter[i];
		csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i - 1];
		old_val = new_val;
	}

	nnz = csrRowPtrA_counter[nrows];
	rows = (uint *)malloc((nrows + 1) * sizeof(uint));
	memcpy(rows, csrRowPtrA_counter, (nrows + 1) * sizeof(uint));
	memset(csrRowPtrA_counter, 0, (nrows + 1) * sizeof(uint));

	cols = (uint *)malloc((nnz) * sizeof(uint));
	memset(cols, 0, (nnz) * sizeof(uint));
	vals = (ValueType *)malloc((nnz) * sizeof(ValueType));
	memset(vals, 0, (nnz) * sizeof(ValueType));

	if (isSymmetric)
	{
		for (uint i = 0; i < nnzA_mtx_report; i++)
		{
			if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
			{
				uint offset = rows[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
				cols[offset] = csrColIdxA_tmp[i];
				vals[offset] = csrValA_tmp[i];
				csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;

				offset = rows[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
				cols[offset] = csrRowIdxA_tmp[i];
				vals[offset] = csrValA_tmp[i];
				csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
			}
			else
			{
				uint offset = rows[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
				cols[offset] = csrColIdxA_tmp[i];
				vals[offset] = csrValA_tmp[i];
				csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
			}
		}
	}
	else
	{
		for (uint i = 0; i < nnzA_mtx_report; i++)
		{
			uint offset = rows[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
			cols[offset] = csrColIdxA_tmp[i];
			vals[offset] = csrValA_tmp[i];
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
#endif