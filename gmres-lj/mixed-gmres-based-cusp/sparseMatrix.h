#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_fp16.h>
extern "C"
{
#include "mmio.h"
}
class SpMH
{
public:
    unsigned int nrows;
    unsigned int ncols;
    unsigned int nnz;
    unsigned int *rows;
    unsigned int *cols;
    half *vals;
    SpMH() : nrows(0), ncols(0), nnz(0), rows(nullptr), cols(nullptr), vals(nullptr){};
};
class SpMS
{
public:
    unsigned int nrows;
    unsigned int ncols;
    unsigned int nnz;
    unsigned int *rows;
    unsigned int *cols;
    float *vals;
    SpMS() : nrows(0), ncols(0), nnz(0), rows(nullptr), cols(nullptr), vals(nullptr){};
};

class SpM
{
public:
    unsigned int nrows;
    unsigned int ncols;
    unsigned int nnz;
    unsigned int *rows;
    unsigned int *cols;
    double *vals;
    SpM() : nrows(0), ncols(0), nnz(0), rows(nullptr), cols(nullptr), vals(nullptr){};
    int readMtx(char *filename);
};

int SpM::readMtx(char *filename)
{
    int ret_code = 0;
    MM_typecode matcode;

    FILE *f = NULL;
    unsigned int nnzA_mtx_report = 0;
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

    unsigned int *csrRowPtrA_counter = (unsigned int *)malloc((nrows + 1) * sizeof(unsigned int));
    memset(csrRowPtrA_counter, 0, (nrows + 1) * sizeof(unsigned int));

    unsigned int *csrRowIdxA_tmp = (unsigned int *)malloc(nnzA_mtx_report * sizeof(unsigned int));
    memset(csrRowIdxA_tmp, 0, nnzA_mtx_report * sizeof(unsigned int));
    unsigned int *csrColIdxA_tmp = (unsigned int *)malloc(nnzA_mtx_report * sizeof(unsigned int));
    memset(csrColIdxA_tmp, 0, nnzA_mtx_report * sizeof(int));
    double *csrValA_tmp = (double *)malloc(nnzA_mtx_report * sizeof(double));
    memset(csrValA_tmp, 0.0, nnzA_mtx_report * sizeof(double));

    for (unsigned int i = 0; i < nnzA_mtx_report; i++)
    {
        unsigned int idxi = 0, idxj = 0;
        double fval = 0.0;
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
        for (unsigned int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++; //对称后，j变成i
        }
    }

    // exclusive scan for csrRowPtrA_counter
    unsigned int old_val = 0, new_val = 0;

    old_val = csrRowPtrA_counter[0];
    csrRowPtrA_counter[0] = 0;
    for (unsigned int i = 1; i <= nrows; i++)
    {
        new_val = csrRowPtrA_counter[i];
        csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i - 1];
        old_val = new_val;
    }

    nnz = csrRowPtrA_counter[nrows];
    rows = (unsigned int *)malloc((nrows + 1) * sizeof(unsigned int));
    memcpy(rows, csrRowPtrA_counter, (nrows + 1) * sizeof(unsigned int));
    memset(csrRowPtrA_counter, 0, (nrows + 1) * sizeof(unsigned int));

    cols = (unsigned int *)malloc((nnz) * sizeof(unsigned int));
    memset(cols, 0, (nnz) * sizeof(unsigned int));
    vals = (double *)malloc((nnz) * sizeof(double));
    memset(vals, 0, (nnz) * sizeof(double));

    if (isSymmetric)
    {
        for (unsigned int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
            {
                unsigned int offset = rows[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
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
                unsigned int offset = rows[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                cols[offset] = csrColIdxA_tmp[i];
                vals[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < nnzA_mtx_report; i++)
        {
            unsigned int offset = rows[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
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
