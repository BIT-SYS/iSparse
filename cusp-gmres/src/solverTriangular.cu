#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void solverTri_GPU(int Am, int i, double *H, double *s)
{
    if (threadIdx.x == 0)
    {
        for (int j = i; j >= 0; j--)
        {
            s[j] /= H[Am * j + j];
            for (int k = j - 1; k >= 0; k--)
            {
                s[k] -= H[k * Am + j] * s[j];
            }
        }
    }
}

void sovlerTri(int Am, int i, double *H, double *s)
{
    solverTri_GPU<<<1, 1, 0>>>(Am, i, H, s);
}