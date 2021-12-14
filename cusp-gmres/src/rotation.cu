#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cusp/complex.h>
#define THREADS_PER_BLOCK_ROTATION 512

__device__ void applyRotation(double &dx, double &dy, double &cs, double &sn)
{
    double temp = cs * dx + sn * dy;
    dy = -cusp::conj(sn) * dx + cs * dy;
    dx = temp;
}
__device__ void generateRotation(double &dx, double &dy, double &cs, double &sn)
{
    if (dx == double(0))
    {
        cs = double(0);
        sn = double(1);
    }
    else
    {
        double scale = cusp::abs(dx) + cusp::abs(dy);
        double norm = scale * std::sqrt(cusp::abs(dx / scale) * cusp::abs(dx / scale) +
                                        cusp::abs(dy / scale) * cusp::abs(dy / scale));
        double alpha = dx / cusp::abs(dx);
        cs = cusp::abs(dx) / norm;
        sn = alpha * cusp::conj(dy) / norm;
    }
}
__global__ void rotation_GPU(uint Am, double *H, double *cs, double *sn, double *s, uint i)
{

    if (threadIdx.x == 0)
    {

        for (uint k = 0; k < i; k++)
        {
            applyRotation(H[k * Am + i], H[(k + 1) * Am + i], cs[k], sn[k]);
        }
        // printf("1 rotation  %le\t%le\t%le\t%le\n", H[i * Am + i], H[(i + 1) * Am + i], cs[i], sn[i]);
        generateRotation(H[i * Am + i], H[(i + 1) * Am + i], cs[i], sn[i]);
        // printf("2 rotation  %le\t%le\t%le\t%le\n",H[i * Am + i], H[(i + 1) * Am + i], cs[i], sn[i]);
        applyRotation(H[i * Am + i], H[(i + 1) * Am + i], cs[i], sn[i]);
        // printf("3 rotation  %le\t%le\t%le\t%le\n", H[i * Am + i], H[(i + 1) * Am + i], cs[i], sn[i]);
        // printf("4 rotation  %le\t%le\t%le\t%le\n", s[i], s[i + 1], cs[i], sn[i]);
        applyRotation(s[i], s[i + 1], cs[i], sn[i]);
        // printf("5 rotation  %le\t%le\t%le\t%le\n", s[i], s[i + 1], cs[i], sn[i]);
    }
}

void rotation2(uint Am, double *H, double *cs, double *sn, double *s, uint i)
{
    rotation_GPU<<<1, 1, 0>>>(Am, H, cs, sn, s, i);
}

__global__ void rotation3_GPU(uint Am, double *H, double *cs, double *sn, double *s, uint i)
{

    uint j;
    __shared__ double si;
    __shared__ double ci;
    if (threadIdx.x == 0)
    {
        double tmp;
        double tmp2;

        double scale;

        if (fabs(H[Am * i + i]) > fabs(H[Am * (i + 1) + i]))
        {
            scale = H[(i + 1) * Am + i] / H[i * Am + i];
            tmp = 1.0 / sqrt(1 + scale * scale);
            cs[i] = tmp;
            sn[i] = scale * tmp;
        }
        else
        {
            scale = (H[i * Am + i]) / (H[(i + 1) * Am + i]);
            sn[i] = 1.0 / sqrt(1 + pow(scale, 2));
            cs[i] = sn[i] * scale;
        }

        tmp = cs[i] * H[i * Am + i] + H[(i + 1) * Am + i] * sn[i];
        tmp2 = cs[i] * H[(i + 1) * Am + i] + H[i * Am + i] * (-1.0 * sn[i]);

        H[(i + 1) * Am + i] = 0;
        H[i * Am + i] = tmp;

        tmp = cs[i] * s[i];
        // printf("0          %f\n", s[i]);
        s[i + 1] = s[i] * (-sn[i]);
        s[i] = tmp;
        // printf("1  %f  %f %f  %f\n", cs[i], sn[i], s[i], s[i + 1]);
    }
}

void rotation3(uint Am, double *H, double *cs, double *sn, double *s, uint i)
{
    rotation3_GPU<<<1, 1, 0>>>(Am, H, cs, sn, s, i);
}