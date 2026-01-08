#include "kernel.h"

__global__ void vecAddKernel(float *dA, float *dB, float *dC, int n){
    
    //Computation of index
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    //Addition of respective elements from A and B
    if(index<n)
        dC[index] = dA[index] + dB[index]; 
}



void launchKernel(float *dA, float *dB, float *dC, int n){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //Kernel launch
    vecAddKernel<<<ceil(n/256.0), 256>>>(dA, dB, dC, n);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float elapsedTime= 0.0f;
    cudaEventElapsedTime(&elapsedTime,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout<<"GPU execution time: "<<elapsedTime<<" ms\n";
}