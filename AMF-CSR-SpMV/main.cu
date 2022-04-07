#include "timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "amf_csr.h"
#include <cusp/io/matrix_market.h>

int main(int argc, char *argv[]) {
    if (argc < 2)  {
        printf("Usage: ./amf_csr mtxFile\n");
        return 0;
	}

    printf("\n=========START=========\n\n");

	csrHostMatrix A;
	cusp::io::read_matrix_market_file(A, argv[1]);

	std::cout << "Read matrix (" << argv[1] << ") ";
	std::cout << "with shape ("  << A.num_rows << "," << A.num_cols << ") and "
              << A.num_entries << " entries" << "\n\n";

 	valHostArray x(A.num_cols);
 	valHostArray y(A.num_rows);
 	for(int i = 0; i < A.num_cols; i++) x[i] = (int(i % 21) - 10);
 	for(int i = 0; i < A.num_rows; i++) y[i] = 0;

	amf_csr_spmv(A, x, y, argv[1]);

    printf("\n=========END=========\n\n");
    return 0;
}