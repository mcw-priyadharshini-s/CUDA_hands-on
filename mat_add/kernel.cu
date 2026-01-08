#include "kernel.h"

__global__ void mat_add(float *arr1, float *arr2, float* res, int m, int n){
    
    // calculating the global index of the element 
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    // checking the boundary conditions
    if (row>=m || col>=n)
        return;

    int index = row*n+col;

    res[index] = arr1[index]+arr2[index];
    
}

void mat_add_launch(float *arr1, float *arr2, float* res, int m, int n){
    int x_dim = 16;
    int y_dim = 16;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // defining the number of threads per block
    dim3 threadsPerBlock(x_dim, y_dim, 1);
    
    // defining the number of blocks per grid
    dim3 blocksPerGrid((n-1)/threadsPerBlock.x+1, (m-1)/threadsPerBlock.y+1, 1);

    cudaEventRecord(start);
    // kernel launch
    mat_add<<<blocksPerGrid, threadsPerBlock>>>(arr1, arr2, res, m, n);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float elapsedTime= 0.0f;
    cudaEventElapsedTime(&elapsedTime,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout<<"gpu time: "<<elapsedTime<<" ms\n";
}
