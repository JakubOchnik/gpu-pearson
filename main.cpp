
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include"structures.h"
#include"parameters.h"
#include"pearsonCorellation.h"

int main()
{
    //main array of function's points
    float x[len] = { 43.0,21.0,25.0,42.0,57.0,59.0 };
    float y[len] = { 99.0,65.0,79.0,75.0,87.0,81.0 };

    float result = pearsonCorellation(x, y);
    printf("Result = %f", result);

    return 0;
}