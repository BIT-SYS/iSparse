all: spmv

#GLOBAL_PARAMETERS
VALUE_TYPE = float
NUM_RUN = 100

#CUDA_PARAMETERS
ARCH = 61
CUDA_CC = nvcc
NVCC_FLAGS = -O3 -w -m64  -Xcompiler -fopenmp -gencode=arch=compute_$(ARCH),code=sm_$(ARCH) -gencode=arch=compute_$(ARCH),code=compute_$(ARCH)
CUDA_INSTALL_PATH = /usr/local/cuda-10.0
CUDA_INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I/home/gjh/NVIDIA_CUDA-10.0_Samples/common/inc 
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcusparse -lgomp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml

TaiChi.o: TaiChi.cu
	$(CUDA_CC) $(NVCC_FLAGS) -o TaiChi.o -c TaiChi.cu $(CUDA_INCLUDES) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
main.o: main.cu
	$(CUDA_CC) -ccbin g++ $(NVCC_FLAGS) -o main.o -c main.cu  $(CUDA_INCLUDES) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
morohutils.o: morohutils.cu
	$(CUDA_CC) -ccbin g++ $(NVCC_FLAGS) -o morohutils.o -c morohutils.cu $(CUDA_INCLUDES) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
mmio.o: mmio.cpp
	$(CUDA_CC) -ccbin g++ $(NVCC_FLAGS) -o mmio.o -c mmio.cpp $(CUDA_INCLUDES) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
spmv: TaiChi.o main.o morohutils.o mmio.o
	$(CUDA_CC) $(NVCC_FLAGS) TaiChi.o main.o morohutils.o mmio.o -o spmv $(CUDA_INCLUDES) $(CUDA_LIBS) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
clean:
	rm *.o 
