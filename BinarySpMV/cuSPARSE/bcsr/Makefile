all : kernel 

ARCH = 70
CUDA_INSTALL_PATH=/usr/local/cuda-10.0
mmio.o: mmio.cpp
	$(CUDA_INSTALL_PATH)/bin/nvcc -O2 -ccbin g++ -I"$(CUDA_INSTALL_PATH)/include" -m64 -gencode arch=compute_${ARCH},code=sm_${ARCH} -o mmio.o -c mmio.cpp
kernel.o: kernel.cu
	$(CUDA_INSTALL_PATH)/bin/nvcc -O2 -ccbin g++ -I"$(CUDA_INSTALL_PATH)/include" -m64 -gencode arch=compute_${ARCH},code=sm_${ARCH} -o kernel.o -c kernel.cu
kernel: mmio.o kernel.o
	$(CUDA_INSTALL_PATH)/bin/nvcc -O2 -ccbin g++ -m64 -gencode arch=compute_${ARCH},code=sm_${ARCH} -o kernel mmio.o kernel.o -lcusparse


clean:
	rm *.o 
