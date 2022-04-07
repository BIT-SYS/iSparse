#include <cusp/csr_matrix.h>
#include <cusp/blas/blas.h>

#ifndef VALUE_TYPE
#define VALUE_TYPE float

typedef unsigned int IndexType;

typedef typename cusp::csr_matrix<IndexType, VALUE_TYPE, cusp::host_memory> csrHostMatrix;
typedef typename cusp::array1d<VALUE_TYPE, cusp::host_memory> valHostArray;

typedef typename cusp::csr_matrix<IndexType, VALUE_TYPE, cusp::device_memory> csrDeviceMatrix;
typedef typename cusp::array1d<VALUE_TYPE, cusp::device_memory> valDeviceArray;

#define FOLDGRAIN 4	// row partition grain
#define MAX_FOLD_ROWS 4
#define FOLD_BITS 2
#define FOLD_MASK 3
// SHARED_MEM_SIZE >= blockDim.x * FOLDGRAIN
#define SHARED_MEM_SIZE 1024 

// record some global information
struct RECORD{
    IndexType num_blocks;
    IndexType num_fold_rows;
    IndexType ave_nnz_in_fold_Rows;
    IndexType TPV;
    IndexType maxNNZ;
    double error;
    double spmv_time;
	double cost_time;
};

#endif

void amf_csr_spmv(csrHostMatrix &A_host, valHostArray &host_x, valHostArray &host_y, char *filePath);
void output(csrHostMatrix &A, csrHostMatrix &mergedA, char *filePath, struct RECORD record1);
void check_spmv(struct RECORD* record1, csrHostMatrix &A_host, csrHostMatrix &mergedA_host, valHostArray &host_x, valHostArray &host_y);
void time_spmv(struct RECORD* record1, csrHostMatrix &A_host, csrHostMatrix &mergedA_host, valHostArray &host_x, valHostArray &host_y);
void generate_mergedA_row_offsets(csrHostMatrix &A, csrHostMatrix &mergedA, struct RECORD* record1);
void generate_mergedA_column_and_value(csrHostMatrix &A, csrHostMatrix &mergedA, struct RECORD* record1);
void RFCOC_spmv_prepare0(struct RECORD* record1, csrDeviceMatrix &A, valDeviceArray &x, valDeviceArray &y);