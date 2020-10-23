#include "utils.h"
#include<stdio.h>
#include<cstdlib>

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