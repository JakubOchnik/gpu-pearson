
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "utils.h"
#include "parameters.h"
#include "pearsonCorellation.h"

__global__ void calculateComponents(PearsonArray mainArr, Components out) {
    __shared__ float cache[threadsPerBlock], cache_s1[threadsPerBlock], cache_s2[threadsPerBlock], cache_s3[threadsPerBlock], cache_s4[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0.0, temps1 = 0.0, temps2 = 0.0,temps3 = 0.0,temps4 = 0.0;
    while (tid < len) {
        temp += mainArr.x[tid] * mainArr.y[tid];
        temps1 += mainArr.x[tid] * mainArr.x[tid];
        temps2 += mainArr.y[tid] * mainArr.y[tid];
        temps3 += mainArr.x[tid];
        temps4 += mainArr.y[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;
    cache_s1[cacheIndex] = temps1;
    cache_s2[cacheIndex] = temps2;
    cache_s3[cacheIndex] = temps3;
    cache_s4[cacheIndex] = temps4;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
            cache_s1[cacheIndex] += cache_s1[cacheIndex + i];
            cache_s2[cacheIndex] += cache_s2[cacheIndex + i];
            cache_s3[cacheIndex] += cache_s3[cacheIndex + i];
            cache_s4[cacheIndex] += cache_s4[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        out.xy[blockIdx.x] = cache[0];
        out.x_sq[blockIdx.x] = cache_s1[0];
        out.y_sq[blockIdx.x] = cache_s2[0];
        out.x[blockIdx.x] = cache_s3[0];
        out.y[blockIdx.x] = cache_s4[0];
    }
}

float pearsonCorellation(const float* x, const float* y)
{
    //structure with points
    PearsonArray p(x, y);
    //temporary sum handlers
    Components temps(len), d_temps(len);

    //device arrays for storing x,y values
    float* dev_a;
    float* dev_b;
    // device arrays for sums xy, x^2, y^2, x, y
    float* dev_xy;
    float* dev_x_sq;
    float* dev_y_sq;
    float* dev_x;
    float* dev_y;

    // -- memory allocation --

    //mainArr, x, y
    cudaMalloc((void**)&dev_a, len * sizeof(float));
    cudaMalloc((void**)&dev_b, len * sizeof(float));
    //allocate all outbound arrays
    cudaMalloc((void**)&dev_xy, blocksPerGrid * sizeof(float));
    cudaMalloc((void**)&dev_x_sq, blocksPerGrid * sizeof(float));
    cudaMalloc((void**)&dev_y_sq, blocksPerGrid * sizeof(float));
    cudaMalloc((void**)&dev_x, blocksPerGrid * sizeof(float));
    cudaMalloc((void**)&dev_y, blocksPerGrid * sizeof(float));

    //copy mainArr to kernel
    cudaMemcpy(dev_a, x, len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, y, len * sizeof(float), cudaMemcpyHostToDevice);
    p.x = dev_a;
    p.y = dev_b;

    d_temps.xy = dev_xy;
    d_temps.x = dev_x;
    d_temps.y = dev_y;
    d_temps.x_sq = dev_x_sq;
    d_temps.y_sq = dev_y_sq;

    //execute the kernel
    calculateComponents<<<blocksPerGrid,threadsPerBlock>>>(p,d_temps);

    //copy arrays of equation components from device to host
    cudaMemcpy(temps.xy, dev_xy, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(temps.x_sq, dev_x_sq, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(temps.y_sq, dev_y_sq, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(temps.x, dev_x, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(temps.y, dev_y, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    //sum all reduced arrays using CPU
    sumOutboundArrays(&temps);

    //print values of components
#if PRINT_COMPONENTS
    printComponents(&temps);
#endif

    //substitute all components into the final Pearson correlation coefficient formula
    float result = substituteIntoFormula(&temps);
    
    //free allocated memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_xy);
    cudaFree(dev_x_sq);
    cudaFree(dev_y_sq);

    temps.freeMem();

    return result;
}