#include <stdio.h>
#include "sparseMatrix.h"
#include <assert.h>
#include <cuda_runtime.h>
#include "spmv.h"
#include "cublas_v2.h"
#include <cusp/monitor.h>

#define THREADS_PER_BLOCK 512

#ifndef VALUETYPE
#define VALUETYPE double
#endif
/*
Gmres(restart)
*/
extern void axpy2(uint Am, double *a, double *b, double s);
extern void scal2(uint Am, double sig, double *a, double *s);
extern void copy2(uint Am, double *a, double *b);
extern void fill2(uint Am, double *a, double s);
extern void rotation2(uint Am, double *H, double *cs, double *sn, double *s, uint i);
extern void sovlerTri(int Am, int i, double *H, double *s);
extern void norm2(uint n, double *a, double *ret);
extern void assign2(double *a, double *s);

void gmres(SpM<double> *A, double *x, double *b, uint restart) //传入的都是cpu上的数据
{
    /*如果表达式 expression 的值为假（即为 0），
    那么它将首先向标准错误流 stderr 打印一条出错信息，
    然后再通过调用 abort 函数终止程序运行；否则，assert 无任何作用。*/

    //超定：M>N  欠定：M<N
    assert(A->nrows == A->ncols);

    SpM<double> A_dev;
    A_dev.matrix_host2device(*A); //将A从CPU上拷到GPU上

    double *x_dev; // x向量在GPu上的内存
    cudaMalloc((void **)&x_dev, (A_dev.nrows + 1) * sizeof(double));
    cudaMemcpy(x_dev, x, A->nrows * sizeof(double), cudaMemcpyHostToDevice);

    double *b_dev; // b在GPU上的内存
    cudaMalloc((void **)&b_dev, (A_dev.nrows + 1) * sizeof(double));
    cudaMemcpy(b_dev, b, A->nrows * sizeof(double), cudaMemcpyHostToDevice);

    double *r0_dev; // n维向量r0
    cudaMalloc((void **)&r0_dev, (A_dev.nrows + 1) * sizeof(double));

    double *V_dev; // V_dev[restart+1][A_dev.nrows] 用来存krylov子空间的单位向量
    cudaMalloc((void **)&V_dev, A_dev.nrows * (restart + 1) * sizeof(double));

    double *s_dev; //残差向量
    cudaMalloc((void **)&s_dev, restart * sizeof(double));

    double *V0;
    cudaMalloc((void **)&V0, A_dev.nrows * sizeof(double));

    double *H; // H[restart+1][restart]
    cudaMalloc((void **)&H, (restart + 1) * restart * sizeof(double));

    double H_cpu[1];

    cusp::array1d<double, cusp::host_memory> resid(1); // monitor判断收敛需要的数据结构
    int i, j, k;

    cublasHandle_t handle;
    cublasCreate(&handle);

    double *cs, *sn; // Givens矩阵
    cudaMalloc((void **)&cs, sizeof(double) * restart);
    cudaMalloc((void **)&sn, sizeof(double) * restart);

    double *beta;
    cudaMalloc((void **)&beta, sizeof(double));

    cusp::array1d<float, cusp::device_memory> monitor_b(A_dev.nrows, 1);
    cusp::monitor<double> monitor(monitor_b, 100, 1e-6, 0, true); // tolerance = 1e-6

    double beta_cpu[1];
    do
    { // cpu 和kernel之间异步执行
        // compute initial residual and its norm
        spmv(&A_dev, x_dev, r0_dev); // r0=Ax
        axpy2(A_dev.nrows, b_dev, r0_dev, -1.0); // r0=b-r0
        norm2(A_dev.nrows, r0_dev, beta); // norm(r0)
        cudaMemcpy(beta_cpu, beta, sizeof(double), cudaMemcpyDeviceToHost);
        scal2(A_dev.nrows, -1.0, r0_dev, beta); // r0/norm(r0)单位化r向量得到v1
        copy2(A_dev.nrows, V_dev, r0_dev); // v_dev[0]=v1

        fill2(A_dev.nrows, s_dev, 0.0); //初始化残差向量
        assign2(s_dev, beta); //残差向量=norm(r)e1

        i = -1;

        do
        {
            i++;
            ++monitor;
            spmv(&A_dev, r0_dev, V0);

            for (k = 0; k <= i; k++)
            {
                cublasDdot(handle, A_dev.nrows, V0, 1, V_dev + k * A_dev.nrows, 1, H + k * restart + i);
                cudaMemcpy(H_cpu, H + k * restart + i, sizeof(double), cudaMemcpyDeviceToHost);
                axpy2(A_dev.nrows, V0, V_dev + A_dev.nrows * k, -H_cpu[0]);
            }
            norm2(A_dev.nrows, V0, H + (i + 1) * restart + i);
            scal2(A_dev.nrows, 1.0, V0, H + (i + 1) * restart + i);
            copy2(A_dev.nrows, V_dev + A_dev.nrows * (i + 1), V0);
            rotation2(restart, H, cs, sn, s_dev, i);

            cudaMemcpy(H_cpu, s_dev + i + 1, sizeof(double), cudaMemcpyDeviceToHost);
            resid[0] = fabs(H_cpu[0] / beta_cpu[0]);
            if (monitor.finished(resid))//修改了cusp对应的误差收敛判断
            {
                i = restart;
                break;
            }
        } while (i + 1 < restart && monitor.iteration_count() + 1 <= monitor.iteration_limit());

        // slove upper triangular system
        // update the solution
        sovlerTri(restart, i, H, s_dev);
        for (j = 0; j <= i; j++)
        {
            cudaMemcpy(H_cpu, s_dev + j, sizeof(double), cudaMemcpyDeviceToHost);
            axpy2(A_dev.nrows, V_dev + j * A_dev.nrows, x_dev, H_cpu[0]);
        }

    } while (!monitor.finished(resid));
    cudaMemcpy(x, x_dev, A_dev.nrows * sizeof(double), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(x_dev);
    cudaFree(A_dev.rows);
    cudaFree(A_dev.cols);
    cudaFree(A_dev.vals);
    cudaFree(s_dev);
    cudaFree(V_dev);
    cudaFree(H);
    cudaFree(V0);
    cudaFree(r0_dev);
    cudaFree(b_dev);
    cudaFree(sn);
    cudaFree(cs);
}