#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "amf_csr.h"
#include "timer.h"
#include <cusp/blas/blas.h>
#include <cusp/multiply.h>
#include <cusp/system/cuda/detail/multiply/coo_flat_k.h>
#include <cusp/system/cuda/detail/multiply/csr_scalar.h>
#include "utility.h"
#include "omp.h"
#include <algorithm>

using namespace std;

typedef unsigned int IndexType;
#define NUM_LOOPS 200

void amf_csr_spmv(csrHostMatrix &A, valHostArray &x, valHostArray &y, char *filePath) {

    csrHostMatrix mergedA(A.num_rows * 2, A.num_cols, A.num_entries);

    struct RECORD record1 = {0};

    // format conversion
    printf("****FORMAT CONVERSION\n");
    generate_mergedA_row_offsets(A, mergedA, &record1);       // generate mergedA.row_offsets
    generate_mergedA_column_and_value(A, mergedA, &record1);  // generate mergedA.column_indices and mergedA.values    
    printf("num_fold_rows=%d num_blocks=%d\n\n", record1.num_fold_rows, record1.num_blocks);

    // parameter setting
    record1.TPV = ceil(sqrt(record1.ave_nnz_in_fold_Rows));
    
    // check SpMV result
    printf("****CHECK SPMV RESULT\n");
    check_spmv(&record1, A, mergedA, x, y);
    printf("error = %f\n\n", record1.error);

    // time SpMV    
    printf("****TIME SPMV\n");
    time_spmv(&record1, A, mergedA, x, y);
    printf("spmv_time = %.6f\n\n", record1.spmv_time);

    // write result to file
    // output(A, mergedA, filePath, record1);
}

void output(csrHostMatrix &A, csrHostMatrix &mergedA, char *filePath, struct RECORD record1)
{
 	char outputName[100] = "output.txt";
   	FILE *fp = fopen(outputName, "a+");
 	if (fp != NULL) {
 		char ch=fgetc(fp);
 		if (ch == EOF) {// file is empty 
 			fprintf(fp, "Matrix M N nnz millisec_spmv\n");
 		}
 	}
 	else {
 		printf("open file failed\n");
 	}
	char temp[100] = {0};
	strcpy(temp, filePath);
 	char *mtxName = strrchr(temp, '/');
 	int len = strlen(mtxName);
 	mtxName[len-4] = '\0';
   	fprintf(fp, "%s %.6f\n", mtxName+1, record1.spmv_time);
   	fclose(fp);
}

void generate_mergedA_row_offsets(csrHostMatrix &A, csrHostMatrix &mergedA, struct RECORD* record1)
{
    // get number of long rows and number of nonzeros in long rows
    IndexType nnz_long_rows = 0, num_long_rows = 0;
    for (IndexType row_id = 0; row_id < A.num_rows; row_id ++)
    {
        IndexType nnz_in_row = A.row_offsets[row_id+1] - A.row_offsets[row_id];
        if (nnz_in_row >= SHARED_MEM_SIZE) // long rows
        {
            nnz_long_rows += nnz_in_row;
            num_long_rows ++;
        }
    }

    // compute an approximate number of nonzeros to guide folding (T_s)
    IndexType ave_nnz_in_fold_Rows = 0; // approxiamete average nnz in folded rows
    if (num_long_rows > 0)  // irregular matrix
    {
        IndexType num_short_rows = A.num_rows - num_long_rows;
        IndexType nnz_short_rows = A.num_entries - nnz_long_rows;
        IndexType appro_num_fold_sRows = (num_short_rows - FOLDGRAIN + 1) / FOLDGRAIN; // approximate number of folded short rows
        ave_nnz_in_fold_Rows = (nnz_short_rows - appro_num_fold_sRows + 1) / appro_num_fold_sRows;
    }
    else    // regular matrix
    {
        IndexType appro_num_fold_rows = (A.num_rows - FOLDGRAIN + 1) / FOLDGRAIN;
        ave_nnz_in_fold_Rows = (A.num_entries - appro_num_fold_rows + 1) / appro_num_fold_rows; 
    }
    record1->ave_nnz_in_fold_Rows = ave_nnz_in_fold_Rows;

    // compute mergedA.row_offsets
    IndexType nnz_in_current_folded_row = 0, nr_in_current_folded_row = 0;
    IndexType fold_row_id = 0;
    mergedA.row_offsets[fold_row_id++] = 0;
    mergedA.row_offsets[fold_row_id++] = 0;
    if (num_long_rows > 0)  // irregular matrix
    {
        for (IndexType row_id = 0; row_id < A.num_rows; row_id ++)
        {
            IndexType nnz_in_row = A.row_offsets[row_id+1] - A.row_offsets[row_id];
            if (nnz_in_row < SHARED_MEM_SIZE) // short rows
            {
                nnz_in_current_folded_row += nnz_in_row;
                nr_in_current_folded_row ++;
                if (nnz_in_current_folded_row > ave_nnz_in_fold_Rows || nr_in_current_folded_row == MAX_FOLD_ROWS)
                {
                    // perform folding
                    if (nnz_in_current_folded_row > SHARED_MEM_SIZE)
                    {
                            mergedA.row_offsets[fold_row_id++] = row_id; // row index
                            mergedA.row_offsets[fold_row_id++] = A.row_offsets[row_id]; // row pointer        
                            row_id--;
                    }
                    else
                    {
                        mergedA.row_offsets[fold_row_id++] = row_id+1; // row index
                        mergedA.row_offsets[fold_row_id++] = A.row_offsets[row_id+1]; // row pointer
                    }
                    nnz_in_current_folded_row = 0;
                    nr_in_current_folded_row = 0;
                }
            }
            else
            {
                if (nnz_in_current_folded_row > 0 || nr_in_current_folded_row > 0)
                {
                    // perform folding
                    mergedA.row_offsets[fold_row_id++] = row_id; // row index
                    mergedA.row_offsets[fold_row_id++] = A.row_offsets[row_id]; // row pointer
                    nnz_in_current_folded_row = 0;
                    nr_in_current_folded_row = 0;
                }
                mergedA.row_offsets[fold_row_id++] = row_id + 1;
                mergedA.row_offsets[fold_row_id++] = A.row_offsets[row_id+1];
            }
        }
        if(nnz_in_current_folded_row > 0) {
            mergedA.row_offsets[fold_row_id++] = A.num_rows;
            mergedA.row_offsets[fold_row_id++] = A.num_entries;
        }
    }
    else    // regular matrix
    {
        for (IndexType row_id = 0; row_id < A.num_rows; row_id++) {
            nnz_in_current_folded_row += A.row_offsets[row_id+1] - A.row_offsets[row_id];
            nr_in_current_folded_row ++;
            if (nnz_in_current_folded_row > ave_nnz_in_fold_Rows || nr_in_current_folded_row == MAX_FOLD_ROWS) {
                mergedA.row_offsets[fold_row_id++] = row_id + 1;
                mergedA.row_offsets[fold_row_id++] = A.row_offsets[row_id+1];
                nnz_in_current_folded_row = 0;
                nr_in_current_folded_row = 0;
            }
        }
        if (nnz_in_current_folded_row > 0)
        {
            mergedA.row_offsets[fold_row_id++] = A.num_rows;
            mergedA.row_offsets[fold_row_id++] = A.num_entries;
        }
    }

    IndexType num_fold_rows = fold_row_id/2 - 1;
    record1->num_fold_rows = num_fold_rows;

    // compute row_blocks
    IndexType nnz_in_current_block = 0;
    IndexType block_id = fold_row_id;
    mergedA.row_offsets[block_id] = 0;
    if (num_long_rows > 0)  // irregular matrix
    {
        for (IndexType row_id = 0; row_id < num_fold_rows; row_id++)
        {
            IndexType nnz_in_row = mergedA.row_offsets[row_id*2+3] - mergedA.row_offsets[row_id*2+1];
            nnz_in_current_block += nnz_in_row;
            if (nnz_in_current_block > SHARED_MEM_SIZE)
            {
                IndexType rows_in_block = row_id - mergedA.row_offsets[block_id];
                if (rows_in_block > 0)
                    mergedA.row_offsets[++block_id] = row_id--;
                else
                    mergedA.row_offsets[++block_id] = row_id + 1;
                nnz_in_current_block = 0;
            }
        }
        if (nnz_in_current_block > 0)
            mergedA.row_offsets[++block_id] = num_fold_rows;
    }
    else    // regular matrix
        mergedA.row_offsets[++block_id] = num_fold_rows;
    mergedA.row_offsets.resize(block_id+1);
    record1->num_blocks = block_id - fold_row_id;
}

void generate_mergedA_column_and_value(csrHostMatrix &A, csrHostMatrix &mergedA, struct RECORD* record1)
{
    IndexType num_blocks = record1->num_blocks;
    IndexType row_blocks_start = mergedA.row_offsets.size() - (num_blocks+1);

    for (IndexType block_id = 0; block_id < num_blocks; block_id++)
    {
        IndexType fold_row_start = mergedA.row_offsets[row_blocks_start + block_id];
        IndexType fold_row_stop = mergedA.row_offsets[row_blocks_start + block_id + 1];
        IndexType rows_to_process = fold_row_stop - fold_row_start;

        IndexType rowIdx_start = mergedA.row_offsets[fold_row_start * 2];
        IndexType rowPtr_start = mergedA.row_offsets[fold_row_start * 2 + 1];
        IndexType rowIdx_stop = mergedA.row_offsets[fold_row_start * 2 + 2];
        IndexType rowPtr_stop = mergedA.row_offsets[fold_row_start * 2 + 3];

        IndexType nnz_to_fold = rowPtr_stop - rowPtr_start;
        IndexType rows_to_fold = rowIdx_stop - rowIdx_start;

        if (rows_to_process > 1 || rows_to_fold > 1 || nnz_to_fold < SHARED_MEM_SIZE)
        {
            #pragma omp parallel for
            for (IndexType fold_row_id = fold_row_start; fold_row_id < fold_row_stop; fold_row_id++)
            {
                IndexType mergedA_index = fold_row_id * 2;
                IndexType row_start = mergedA.row_offsets[mergedA_index];
                IndexType row_stop = mergedA.row_offsets[mergedA_index + 2];
                IndexType nz_start = mergedA.row_offsets[mergedA_index + 1];
                IndexType nz_stop = mergedA.row_offsets[mergedA_index + 3];
                IndexType rows_in_fold_row = row_stop - row_start;
                IndexType s_rowPtr_A[rows_in_fold_row];
                IndexType e_rowPtr_A[rows_in_fold_row];
                IndexType cur_rowPtr_A[rows_in_fold_row];
                IndexType tmp_colIdx[rows_in_fold_row];
                IndexType residual = 0;
                for (IndexType ii = 0; ii < rows_in_fold_row; ii++)
                {
                    IndexType real_rowId = row_start + ii;
                    s_rowPtr_A[ii] = A.row_offsets[real_rowId];
                    e_rowPtr_A[ii] = A.row_offsets[real_rowId + 1];
                    cur_rowPtr_A[ii] = s_rowPtr_A[ii];
                    if (cur_rowPtr_A[ii] < e_rowPtr_A[ii])
                    {
                        tmp_colIdx[ii] = A.column_indices[cur_rowPtr_A[ii]];
                        residual++;
                    }
                    else
                        tmp_colIdx[ii] = A.num_cols;
                }

                while (residual > 0)
                {
                    IndexType min_colIdx = A.num_cols; // record the minimum column index of four nonzeros
                    IndexType min_local_rowIdx = 0; // record the corresponding block index of min_colIdx
                    IndexType tmp_collision_num = 0;
                    IndexType tmp_collision_local_rowIdx[rows_in_fold_row];
                    for (IndexType ii = 0; ii < rows_in_fold_row; ii++)
                    {
                        if (tmp_colIdx[ii] < min_colIdx)
                        {
                            tmp_collision_num = 0;
                            min_colIdx = tmp_colIdx[ii];
                            min_local_rowIdx = ii;
                        }
                        if (tmp_colIdx[ii] == min_colIdx && ii > min_local_rowIdx)
                        {
                            tmp_collision_local_rowIdx[tmp_collision_num] = ii;
                            tmp_collision_num++;
                        }
                    }

                    // save the minimum nonzero to merged block
                    IndexType tmp_min_colIdx = (min_colIdx << FOLD_BITS);
                    mergedA.column_indices[nz_start] = min_local_rowIdx; // block information
                    mergedA.column_indices[nz_start] += tmp_min_colIdx;
                    mergedA.values[nz_start++] = 
                        A.values[cur_rowPtr_A[min_local_rowIdx]]; // nonzeros of the
                                                                // array csrVal are
                                                                // also required to be
                                                                // reordered
                    for (IndexType ii = 0; ii < tmp_collision_num; ii++)
                    {
                        mergedA.column_indices[nz_start] = tmp_collision_local_rowIdx[ii];
                        mergedA.column_indices[nz_start] += tmp_min_colIdx;
                        mergedA.values[nz_start++] =
                            A.values[cur_rowPtr_A[tmp_collision_local_rowIdx
                                                    [ii]]]; // nonzeros of the
                                                            // array csrVal are
                                                            // also required to
                                                            // be reordered
                    }

                    // update the tmp_colIdx[min_blockId]
                    for (IndexType ii = min_local_rowIdx; ii < rows_in_fold_row; ii++)
                    {
                        if (tmp_colIdx[ii] == min_colIdx)
                        {
                            if (cur_rowPtr_A[ii] + 1 < e_rowPtr_A[ii])
                                tmp_colIdx[ii] = A.column_indices[++cur_rowPtr_A[ii]];
                            else
                                tmp_colIdx[ii] = A.num_cols;
                        }
                    }

                    residual = 0;
                    for (IndexType ii = 0; ii < rows_in_fold_row; ii++)
                    {
                        if (tmp_colIdx[ii] < A.num_cols)
                            residual++;
                    }
                }
                if (nz_start != nz_stop) printf("ERROR: folding error\n");
            }
        }
        else
        {
            memcpy(&mergedA.column_indices[rowPtr_start], &A.column_indices[rowPtr_start], sizeof(IndexType) * (rowPtr_stop - rowPtr_start));
            memcpy(&mergedA.values[rowPtr_start], &A.values[rowPtr_start], sizeof(VALUE_TYPE) * (rowPtr_stop - rowPtr_start));
        }
    }
}

void check_spmv(struct RECORD* record1, csrHostMatrix &A_host, csrHostMatrix &mergedA_host, valHostArray &host_x, valHostArray &host_y) {
	valHostArray y_correct(host_y.begin(), host_y.end());
	/*+--------------------+
	+  right SpMV in CPU   |
	+---------------------*/
	cusp::multiply<csrHostMatrix, valHostArray, valHostArray>(A_host, host_x, y_correct);

	/*+--------------------------+
	+  AMF-CSR SpMV in GPU  |
	+---------------------------*/
	// copy data from host to device
	csrDeviceMatrix mergedA_device(mergedA_host);
	valDeviceArray device_x(host_x.begin(), host_x.end());
	valDeviceArray device_y(host_y.begin(), host_y.end());

    // SpMV in GPU
    RFCOC_spmv_prepare0(record1, mergedA_device, device_x, device_y);

	// copy device_y to host vector test_y_copy
	valHostArray test_y_copy(device_y.begin(), device_y.end());

	/*+------------------+
	+  Validate result   |
	+-------------------*/
	record1->error = l2_error(A_host.num_rows, thrust::raw_pointer_cast(&test_y_copy[0]), thrust::raw_pointer_cast(&y_correct[0]));
}

void time_spmv(struct RECORD* record1, csrHostMatrix &A_host, csrHostMatrix &mergedA_host, valHostArray &host_x, valHostArray &host_y) {
	// transfer data
	csrDeviceMatrix mergedA_device(mergedA_host);

	valDeviceArray device_x(host_x.begin(), host_x.end());
	valDeviceArray device_y(host_y.begin(), host_y.end());

    // warmup
    for (int i = 0; i < 10; i++)
        RFCOC_spmv_prepare0(record1, mergedA_device, device_x, device_y);
	cudaThreadSynchronize();

    // time several SpMV iterations
	timer t;
	for (int i = 0; i < NUM_LOOPS; i++) {
		RFCOC_spmv_prepare0(record1, mergedA_device, device_x, device_y);
	}
	cudaThreadSynchronize();
    record1->spmv_time = t.seconds_elapsed() *  1e3 / NUM_LOOPS;   // convert to milliseconds
}