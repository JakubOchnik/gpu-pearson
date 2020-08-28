
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include"structures.h"

#define PRINT_COMPONENTS true
//NUMBER OF (X,Y) POINTS
const int len = 6;

#define imin(a,b) (a<b?a:b)
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (len + threadsPerBlock - 1) / threadsPerBlock);

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

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

void sumOutboundArrays(Components* temps) {
    for (int i = 0; i < blocksPerGrid; i++) {
        temps->xy_res += temps->xy[i];
        temps->x_sq_res += temps->x_sq[i];
        temps->y_sq_res += temps->y_sq[i];
        temps->x_res += temps->x[i];
        temps->y_res += temps->y[i];
    }
}

void printComponents(Components* temps) {
    printf("(sigma)xi * yi = %f\n", temps->xy_res);
    printf("(sigma)xi^2 = %f\n", temps->x_sq_res);
    printf("(sigma)yi^2 = %f\n", temps->x_sq_res);
    printf("(sigma)xi = %f\n", temps->x_res);
    printf("(sigma)yi = %f\n", temps->y_res);
}

float substituteIntoFormula(Components* temps) {
    float top = len * temps->xy_res - temps->x_res * temps->y_res;
    float b1 = len * temps->x_sq_res - pow(temps->x_res, 2);
    float b2 = len * temps->y_sq_res - pow(temps->y_res, 2);
    float b_multi = b1 * b2;
    float b_sq = sqrt(b_multi);
    float result = top / b_sq;
    return result;
}

int main()
{
    //main array of function's points
    float x[len] = { 43.0,21.0,25.0,42.0,57.0,59.0 };
    float y[len] = { 99.0,65.0,79.0,75.0,87.0,81.0 };
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
    
    printf("r = %f\n", result);

    //free allocated memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_xy);
    cudaFree(dev_x_sq);
    cudaFree(dev_y_sq);

    temps.freeMem();

    return 0;
}