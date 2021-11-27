# iSparse
## Circuit Simulation
### miniXyce_1.0
Original version from [Mantevo-ROSE](https://github.com/peihunglin/Mantevo-ROSE).
### miniXyce_1.0_vectorCSR
Still read netlist, replace SpMV with vector CSR SpMV implementation.
### miniXyce_1.0_modify
read sparse matrix instead of netlist, replace SpMV with vector CSR SpMV implementation.
### miniXyce_1.0_cusparse
Still read netlist, replace SpMV with new cusparse SpMV API.

> compile and run:
> cd miniXyce_1.0/miniXyce_ref
> make
> ./miniXyce.x --circuit netlistFile/matrixFile --pf params.txt
