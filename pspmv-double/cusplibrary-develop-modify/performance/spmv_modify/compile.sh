/usr/local/cuda-10.0/bin/nvcc -O2 -w -m64 -gencode arch=compute_70,code=sm_70 -I/home/GaoJH/spmv/cusplibrary-develop-modify/ spmv.cu
#/usr/local/cuda-10.0/bin/nvcc -O2 -w -m64 -Xptxas -dlcm=cg -gencode arch=compute_70,code=sm_70 -I/home/GaoJH/spmv/cusplibrary-develop-modify/ spmv.cu
