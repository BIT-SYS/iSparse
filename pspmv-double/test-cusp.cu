#include "cusp/csr_matrix.h"
#include "cusp/io/matrix_market.h"
#include <stdlib.h>
#include "cusp/multiply.h"
#include "cusp/array1d.h"
#include "cusp/array2d.h"
#include "cusp/multiply.h"
#include "cusp/functional.h"

int main(int argc, char *argv[])
{
    // cusp::csr_matrix<int, double, cusp::host_memory> A;

    // load a matrix stored in MatrixMarket format
    // cusp::io::read_matrix_market_file(A, argv[1]);
    // cusp::csr_matrix<IndexType, ValueType, cusp::host_memory> test_matrix_on_host(A);
    // transfer csr_matrix to device
    typedef typename cusp::csr_matrix<int, double, cusp::device_memory> DeviceMatrix;
    typedef typename cusp::array1d<double, cusp::device_memory> DeviceArray;
    // typedef typename cusp::array1d<double, cusp::device_memory> DeviceArray2d;
    DeviceMatrix test_matrix_on_device(A);

    const int M = A.num_rows;
    const int N = A.num_cols;

    // create host input (x) and output (y) vectors
    cusp::array1d<double, cusp::host_memory> host_x(N);
    cusp::array1d<double, cusp::host_memory> host_y(M);
    srand(time(NULL));
    for (int i = 0; i < N; i++)
        host_x[i] = (1 * rand() / (RAND_MAX + 1.0));
    for (int i = 0; i < M; i++)
        host_y[i] = 0;

    // define multiply functors
    cusp::zero_functor<double> initialize;
    thrust::multiplies<double> combine;
    thrust::plus<double>       reduce;

    cusp::multiply(test_matrix_on_device, host_x, host_y,initialize,combine,reduce);

    return 0;
}