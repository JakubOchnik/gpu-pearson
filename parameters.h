#pragma once

//NUMBER OF (X,Y) POINTS
const int len = 6;

//PRINT COMPONENTS (sums)
#define PRINT_COMPONENTS true

//PARAMETERS OF GRID STRUCTURE USED IN KERNEL
#define imin(a,b) (a<b?a:b)
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (len + threadsPerBlock - 1) / threadsPerBlock);

//FIX VS SYNTAX HIGHLIGHTING
#ifdef __INTELLISENSE__
void __syncthreads();
#endif