#pragma once
#include<stdlib.h>
typedef struct arr {
    const float* x;
    const float* y;
    arr(const float orig_x[], const float orig_y[]) {
        x = orig_x;
        y = orig_y;
    }
}PearsonArray;

typedef struct sums {
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
    sums(int len) {
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
    void freeMem() {
        free(xy);
        free(x_sq);
        free(y_sq);
        free(x);
        free(y);
    }
}Components;
