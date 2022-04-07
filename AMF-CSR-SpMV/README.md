# AMF-CSR-SpMV
A cache locality-improved and local-balanced SpMV algorithm. 

## prerequisites
### Linux
- CUDA 10.0

## Usage
- Modify variable ```ARCH``` and ```CUDA_INSTALL_PATH``` in Makefile according to your own GPU and CUDA environment
- Compile and run programs
```shell
make
./amf_csr circuit_1.mtx
```
