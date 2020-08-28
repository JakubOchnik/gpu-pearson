
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<chrono>
#include<iostream>

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

typedef struct arr {
    float* x;
    float* y;
    arr(float orig_x[], float orig_y[]) {
        x = orig_x;
        y = orig_y;
    }
}PearsonArray;

typedef struct coeffs {
    float* xy;
    float* x_sq;
    float* y_sq;
    float* x;
    float* y;
    float xy_res;
    float x_sq_res;
    float y_sq_res;
    float x_res;
    float y_res;
    coeffs() {
        xy = (float*)malloc(len * sizeof(float));
        x_sq = (float*)malloc(len * sizeof(float));
        y_sq = (float*)malloc(len * sizeof(float));
        x = (float*)malloc(len * sizeof(float));
        y = (float*)malloc(len * sizeof(float));
        xy_res = 0.0f;
        x_sq_res = 0.0f;
        y_sq_res = 0.0f;
        x_res = 0.0f;
        y_res = 0.0f;
    }
}sums;


__global__ void dot(PearsonArray mainArr, sums out) {
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


int main()
{
    float* x = (float*)malloc(len * sizeof(float));
    float* y = (float*)malloc(len * sizeof(float));
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
    PearsonArray p(x, y);
    sums temps, d_temps;
    /*for (int i = 0; i < len; i++) {
        x[i] = i;
        y[i] = i * 2;
    }*/
    //array ->x,y
    float* dev_a;
    float* dev_b;
    // temp arrays -> xy, x^2, y^2, x, y
    float* dev_xy;
    float* dev_x_sq;
    float* dev_y_sq;
    float* dev_x;
    float* dev_y;

    //mainArr, x, y
    cudaMalloc((void**)&dev_a, len * sizeof(float));
    cudaMalloc((void**)&dev_b, len * sizeof(float));
    //alloc all out arrays
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


    dot<<<blocksPerGrid,threadsPerBlock>>>(p,d_temps);


    cudaMemcpy(temps.xy, dev_xy, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(temps.x_sq, dev_x_sq, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(temps.y_sq, dev_y_sq, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(temps.x, dev_x, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(temps.y, dev_y, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < blocksPerGrid; i++)
        temps.xy_res += temps.xy[i];
    for (int i = 0; i < blocksPerGrid; i++)
        temps.x_sq_res += temps.x_sq[i];
    for (int i = 0; i < blocksPerGrid; i++)
        temps.y_sq_res += temps.y_sq[i];
    for (int i = 0; i < blocksPerGrid; i++)
        temps.x_res += temps.x[i];
    for (int i = 0; i < blocksPerGrid; i++)
        temps.y_res += temps.y[i];


    std::cout << temps.xy_res << std::endl;
    std::cout << temps.x_sq_res << std::endl;
    std::cout << temps.y_sq_res << std::endl;
    std::cout << temps.x_res << std::endl;
    std::cout << temps.y_res << std::endl;

    float top = len * temps.xy_res - temps.x_res * temps.y_res;
    float b1 = len * temps.x_sq_res - pow(temps.x_res,2);
    float b2 = len * temps.y_sq_res - pow(temps.y_res,2);
    float b_multi = b1 * b2;
    float b_sq = sqrt(b_multi);
    float result = top / b_sq;
    std::cout << result;


    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_xy);
    cudaFree(dev_x_sq);
    cudaFree(dev_y_sq);

    return 0;
}