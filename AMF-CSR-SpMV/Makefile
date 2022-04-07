all: amf_csr

# CUDA PARAMETERS
# please modify ${ARCH} and ${CUDA_INSTALL_PATH} according to your CUDA Capability and CUDA install path
ARCH = 61
NVCC_FLAGS = -O2 -w -m64 -gencode arch=compute_${ARCH},code=sm_${ARCH} -Xcompiler -O2,-fopenmp
CUDA_INSTALL_PATH = /usr/local/cuda-10.0
CUDA_INCLUDES = -I$(CUDA_INSTALL_PATH)/include

kernel.o: kernel.cu
	$(CUDA_INSTALL_PATH)/bin/nvcc $(NVCC_FLAGS) $(CUDA_INCLUDES) -o kernel.o -c kernel.cu
amf_csr.o: amf_csr.cu
	$(CUDA_INSTALL_PATH)/bin/nvcc $(NVCC_FLAGS) $(CUDA_INCLUDES) -o amf_csr.o -c amf_csr.cu
main.o: main.cu
	$(CUDA_INSTALL_PATH)/bin/nvcc $(NVCC_FLAGS) $(CUDA_INCLUDES) -o main.o -c main.cu
amf_csr: kernel.o amf_csr.o main.o
	$(CUDA_INSTALL_PATH)/bin/nvcc $(NVCC_FLAGS) $(CUDA_INCLUDES) -o amf_csr main.o amf_csr.o kernel.o


clean: 
	rm *.o
