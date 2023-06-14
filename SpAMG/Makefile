CUDA_PATH = /usr/local/cuda-11.2
NVCC = $(CUDA_PATH)/bin/nvcc
ARCH = --gpu-architecture=compute_70 --gpu-code=compute_70,sm_70

NVCC_FLAGS = -w \
			-std=c++14 \
			$(ARCH) \
			-DNVCC \
			-O3 \
			-DNDEBUG \
			--Werror cross-execution-space-call \
			-DDISABLE_MIXED_PRECISION \
			-DCUSPARSE_GENERIC_INTERFACES \
			-DCUSPARSE_USE_GENERIC_SPGEMM \
			-DNVTX_RANGES \
			-Xcompiler=-O3 \
			-Xcompiler=-Wno-terminate \
			-Xcompiler=-fopenmp \
			-Xcompiler=-DRAPIDJSON_DEFINED \
			-Xcompiler=-fPIC \
			-Xcompiler=-rdynamic \
			-Xcompiler=-fvisibility=default \
			-lcusparse

--ptxas-options=-v \

NVCC_FLAGS2 = -w \
			-std=c++14 \
			$(ARCH) \
			-DNVCC \
			-O3 \
			-DNDEBUG \
			--Werror cross-execution-space-call \
			-DDISABLE_MIXED_PRECISION \
			-DCUSPARSE_GENERIC_INTERFACES \
			-DCUSPARSE_USE_GENERIC_SPGEMM \
			-DNVTX_RANGES \
			-Xcompiler=-O3 \
			-Xcompiler=-Wno-terminate \
			-Xcompiler=-fopenmp \
			-Xcompiler=-DRAPIDJSON_DEFINED \
			-Xcompiler=-fPIC \
			-Xcompiler=-rdynamic \
			-Xcompiler=-fvisibility=default \
			--fmad=true \
			--use_fast_math \
			-lcusparse 
# --ptxas-options=-v
			

# NVCC_L_FLAGS = $(ARCH)
# EXTRA_LIBS = -lcublas -lcusparse -lcusolver -Xlinker=-rpath=$(CUDA_PATH)/lib64
# -G -Xcompiler=-g

all: origin_aq_bq

origin_aq_bq:
	$(NVCC) $(NVCC_FLAGS2) origin_aq_bq.cu -o origin_aq_bq.out
clean:
	rm *.out