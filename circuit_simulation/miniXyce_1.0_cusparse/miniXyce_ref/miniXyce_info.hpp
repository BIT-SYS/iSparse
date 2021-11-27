#ifndef miniXyce_info_hpp
#define miniXyce_info_hpp

#define MINIXYCE_HOSTNAME "ubun"
#define MINIXYCE_KERNEL_NAME "'Linux'"
#define MINIXYCE_KERNEL_RELEASE "'5.4.0-90-generic'"
#define MINIXYCE_PROCESSOR "'x86_64'"

#define MINIXYCE_CXX "'/usr/local/cuda-11.2/bin/nvcc'"
#define MINIXYCE_CXX_VERSION "'nvcc: NVIDIA (R) Cuda compiler driver'"
#define MINIXYCE_CXXFLAGS "'-O3 -w -m64 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70  -Xcompiler -O3,-funroll-all-loops  -I/home/GaoJH/spmv/cusplibrary  -I/usr/local/cuda-11.2/include -I/usr/local/cuda-11.2/samples/common/inc'"

#endif
