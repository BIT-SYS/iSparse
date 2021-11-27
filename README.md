# iSparse
## Circuit Simulation
Please refer to https://github.com/peihunglin/Mantevo-ROSE/tree/master/miniXyce_1.0 for its detailed introduction.

- **miniXyce_1.0:**
The programs in this directory are original version, without any modifying.

- **miniXyce_1.0_vectorCSR:**
The input of this version still is the netlist file, and the SpMV (sparse matrix-vetor multiplication) implementation is replaced with GPU-based vector CSR SpMV.

- **miniXyce_1.0_modify:**
The input of this version is sparse matrix instread of netlist file, and the SpMV (sparse matrix-vetor multiplication) implementation is replaced with GPU-based vector CSR SpMV.

- **miniXyce_1.0_cusparse:**
The input of this version is the netlist file, and the SpMV (sparse matrix-vetor multiplication) implementation is replaced with new SpMV API in cuSPARSE library.


**Compile and run for program of circuit simulation**

compile:
- > cd miniXyce_1.0xxx/miniXyce_ref
- > make

run:
- > ./miniXyce.x --circuit netlistFile/matrixFile --pf params.txt
