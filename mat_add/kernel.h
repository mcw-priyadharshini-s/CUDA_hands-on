#pragma once

#include <cuda_runtime.h>
#include <iostream>
using  namespace std;
void mat_add_launch(float *arr1, float *arr2, float* res, int m, int n);