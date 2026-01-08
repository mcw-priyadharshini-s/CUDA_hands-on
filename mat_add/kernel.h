#pragma once

#include<iostream>
using namespace std;

#include<cuda_runtime.h>

void launchKernel(float *dA, float *dB, float *dC, int m, int n);