#include "kernel.h"
__global__ void matAddKernel(float *dA, float *dB, float *dC, int m, int n){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int index = row * n + col;

    if(row < m && col < n){
        dC[index] = dA[index] + dB[index];
    }

}

void launchKernel(float *dA, float *dB, float *dC, int m, int n){
    
    int x_dim = 16;
    int y_dim = 16;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Defining number of threads per block
    dim3 threadsPerBlock(x_dim, y_dim, 1);

    //Defining number of blocks per grid
    dim3 blocksPerGrid((n-1)/threadsPerBlock.x+1, (m-1)/threadsPerBlock.y+1, 1);

    cudaEventRecord(start);
    //Kernel launch
    matAddKernel<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, m, n);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float elapsedTime= 0.0f;
    cudaEventElapsedTime(&elapsedTime,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout<<"GPU execution time: "<<elapsedTime<<" ms\n";
}
