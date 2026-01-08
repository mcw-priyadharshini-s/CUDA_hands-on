#pragma once

#include <cuda_runtime.h>
#include <iostream>
#define TILE_WIDTH 16
using  namespace std;
void mat_mul_launch(float *arr1, float *arr2, float* res, int m, int n, int k);