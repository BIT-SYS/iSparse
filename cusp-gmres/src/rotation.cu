#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define THREADS_PER_BLOCK_ROTATION 512

__global__ void rotation_GPU(uint Am, double *H, double *cs, double *sn, double *s, uint i)
{
    double tmp;
    uint j;
    __shared__ double si;
    __shared__ double ci;
    if (threadIdx.x == 0)
    {
        for (uint k = 0; k < i; k++)
        {
            tmp = cs[k] * H[k * Am + i] + H[(k + 1) * Am + i] * sn[k];
            H[(k + 1) * Am + i] = cs[k] * H[(k + 1) * Am + i] + H[k * Am + i] * (-sn[k]);
            H[k * Am + i] = tmp;
        }
        double scale;
        if (fabs(H[Am * i + i]) > fabs(H[Am * (i + 1) + i]))
        {
            scale = pow(H[i * Am + i], 2) + pow(H[(i + 1) * Am + i], 2);
            cs[i] = H[i * Am + i] / sqrt(scale);
            sn[i] = H[(i + 1) * Am + i] / sqrt(scale);
        }
        else
        {
            scale = fabs(H[i * Am + i]) / fabs(H[(i + 1) * Am + i]);
            sn[i] = 1.0 / sqrt(1 + pow(scale, 2));
            cs[i] = sn[i] * scale;
        }

        tmp = cs[i] * H[i * Am + i] + H[(i + 1) * Am + i] * sn[i];
        H[(i + 1) * Am + i] = 0; // cs[i] * H[(i + 1) * Am + i] + H[i * Am + i] * (-sn[i]);
        H[i * Am + i] = tmp;

        tmp = cs[i] * s[i] + s[(i + 1)] * sn[i];
        s[i + 1] = cs[i] * s[i + 1] + s[i] * (-sn[i]);
        s[i] = tmp;
    }
}

void rotation2(uint Am, double *H, double *cs, double *sn, double *s, uint i)
{
    rotation_GPU<<<1, 1, 0>>>(Am, H, cs, sn, s, i);
}