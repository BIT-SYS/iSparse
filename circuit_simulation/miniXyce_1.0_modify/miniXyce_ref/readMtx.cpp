#include "mmio.h"
#include "readMtx.hpp"
#include <iostream>
using namespace std;

mX_linear_DAE* readMtxToDAE(std::string filename, int p, int pid, int &n, int &num_internal_nodes, int &num_voltage_sources, int &num_current_sources, int &num_resistors, int &num_capacitors, int &num_inductors)
{
    mX_linear_DAE* dae = new mX_linear_DAE();
	dae->A = new distributed_sparse_matrix();
	dae->B = new distributed_sparse_matrix();

    distributed_sparse_matrix* A = dae->A;
	distributed_sparse_matrix* B = dae->B;

    IndexType* csrRowPtr, *csrColIdx;
    VALUE_TYPE* csrVal;
    int rows=0, cols=0, nnz=0;

    readMtx(filename.c_str(), rows, cols, nnz, csrRowPtr, csrColIdx, csrVal);
    printf("m = %d, n = %d, nnz = %d\n", rows, cols, nnz);
    n = rows;

    int start_row = (n/p)*(pid) + ((pid < n%p) ? pid : n%p);
	int end_row = start_row + (n/p) - 1 + ((pid < n%p) ? 1 : 0);
    A->start_row = start_row;
	A->end_row = end_row;
    B->start_row = start_row;
	B->end_row = end_row;

	std::cout << "start_row = " << start_row << ", end_row = " << end_row << std::endl;
	// initialize
    for (int i = start_row; i <= end_row; i++)
	{
		distributed_sparse_matrix_entry* null_ptr_1 = 0;
		A->row_headers.push_back(null_ptr_1);
		B->row_headers.push_back(null_ptr_1);

		mX_linear_DAE_RHS_entry* null_ptr_2 = 0;
		(dae->b).push_back(null_ptr_2);
	}

	// copy CSR data to DAE
    for (int i = start_row; i <= end_row; i++) {
        for (int j = csrRowPtr[i-start_row]; j < csrRowPtr[i-start_row+1]; j++)
        {
            int colIdx = csrColIdx[j];
			distributed_sparse_matrix_add_to(A, i-start_row, colIdx, csrVal[j], n, p);
        }
    }

	// prepare RHS
	mX_source* src = new DC(1);
	mX_scaled_source* scaled_src = new mX_scaled_source();
	scaled_src->src = src;
	scaled_src->scale = (double)(1);
    dae->b[start_row] = new mX_linear_DAE_RHS_entry();
	(dae->b[start_row])->scaled_src_list.push_back(scaled_src);
	// for (int i = start_row; i <= end_row; i++)
	// {
	// 	dae->b[i-start_row] = new mX_linear_DAE_RHS_entry();
	// 	(dae->b[i-start_row])->scaled_src_list.push_back(scaled_src);
	// }

    free(csrRowPtr); free(csrColIdx); free(csrVal);
    return dae;
}

int readMtx(const char *filename, int &m, int &n, int &nnzA, IndexType *&csrRowPtrA, IndexType *&csrColIdxA,
	VALUE_TYPE *&csrValA)
{
	int ret_code = 0;
	MM_typecode matcode;

	FILE *f = NULL;
	int nnzA_mtx_report = 0;
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

	IndexType *csrRowPtrA_counter = (IndexType *)malloc((m + 1) * sizeof(IndexType));
	memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(IndexType));

	IndexType *csrRowIdxA_tmp = (IndexType *)malloc(nnzA_mtx_report * sizeof(IndexType));
	memset(csrRowIdxA_tmp, 0, nnzA_mtx_report * sizeof(IndexType));
	IndexType *csrColIdxA_tmp = (IndexType *)malloc(nnzA_mtx_report * sizeof(IndexType));
	memset(csrColIdxA_tmp, 0, nnzA_mtx_report * sizeof(IndexType));
	VALUE_TYPE *csrValA_tmp = (VALUE_TYPE *)malloc(nnzA_mtx_report * sizeof(VALUE_TYPE));
	memset(csrValA_tmp, 0.0, nnzA_mtx_report * sizeof(VALUE_TYPE));

	for (int i = 0; i < nnzA_mtx_report; i++)
	{
		IndexType idxi = 0, idxj = 0;
		double fval = 0.0;
		int ival = 0;

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
		for (int i = 0; i < nnzA_mtx_report; i++) {
			if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
				csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
		}
	}

	// exclusive scan for csrRowPtrA_counter
	IndexType old_val = 0, new_val = 0;

	old_val = csrRowPtrA_counter[0];
	csrRowPtrA_counter[0] = 0;
	for (int i = 1; i <= m; i++)
	{
		new_val = csrRowPtrA_counter[i];
		csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i - 1];
		old_val = new_val;
	}

	nnzA = csrRowPtrA_counter[m];
	csrRowPtrA = (IndexType *)malloc((m + 1) * sizeof(IndexType));
	memcpy(csrRowPtrA, csrRowPtrA_counter, (m + 1) * sizeof(IndexType));
	memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(IndexType));

	csrColIdxA = (IndexType *)malloc(nnzA * sizeof(IndexType));
	memset(csrColIdxA, 0, nnzA * sizeof(IndexType));
	csrValA = (VALUE_TYPE *)malloc(nnzA * sizeof(VALUE_TYPE));
	memset(csrValA, 0, nnzA * sizeof(VALUE_TYPE));

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