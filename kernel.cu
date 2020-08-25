
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif
#include <stdio.h>
#include<stdlib.h>

#define imin(a,b) (a<b?a:b)

const int threadsPerBlock = 256;
const int len = 5;
const int blocksPerGrid = imin(32, (len + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float* a, float* b, float* c) {
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;
    while (tid < len) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        c[blockIdx.x] = cache[0];
    }
}


int main()
{
    float* x = (float*)malloc(len * sizeof(float));
    float* y = (float*)malloc(len * sizeof(float));
    float* p_c = (float*)malloc(blocksPerGrid * sizeof(float));
    x[0] = 3.0;
    y[0] = 20.0;
    x[1] = 3.0;
    y[1] = 25.0;
    x[2] = 2.0;
    y[2] = 20.0;
    x[3] = 4.0;
    y[3] = 30.0;
    x[4] = 1.0;
    y[4] = 10.0;
    float c = 0.0;
    float* dev_a;
    float* dev_b;
    float* dev_p_c;

    cudaMalloc((void**)&dev_a, len * sizeof(float));
    cudaMalloc((void**)&dev_b, len * sizeof(float));
    cudaMalloc((void**)&dev_p_c, blocksPerGrid * sizeof(float));

    cudaMemcpy(dev_a, x, len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, y, len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p_c, p_c, len * sizeof(float), cudaMemcpyHostToDevice);
    return 0;
}