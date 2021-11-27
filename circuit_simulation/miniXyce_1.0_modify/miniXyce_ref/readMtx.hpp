#include "mX_linear_DAE.h"

using namespace mX_linear_DAE_utils;

typedef int IndexType;
typedef double VALUE_TYPE;

mX_linear_DAE* readMtxToDAE(std::string filename, int p, int pid, int &n, int &num_internal_nodes, int &num_voltage_sources, int &num_current_sources, int &num_resistors, int &num_capacitors, int &num_inductors);

int readMtx(const char *filename, int &m, int &n, int &nnzA, IndexType *&csrRowPtrA, IndexType *&csrColIdxA,
	VALUE_TYPE *&csrValA);