#include "kernel.h"

__global__ void mat_mul(float *arr1, float *arr2, float* res, int m, int n, int k){
    
    // calculating the global index of the element
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    //initialising shared memory for tiles
    __shared__ float ds_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_b[TILE_WIDTH][TILE_WIDTH];

    for (int t = 0; t<(n-1)/TILE_WIDTH+1; t++){

        //loading the data from global memory into tiles
        if (row<m && t*TILE_WIDTH+tx<n)
            ds_a[ty][tx] = arr1[row*n+t*TILE_WIDTH+tx];
        else
            ds_a[ty][tx] = 0.0f;

        if (t*TILE_WIDTH+ty<n && col<k)
            ds_b[ty][tx] = arr2[(t*TILE_WIDTH+ty)*k+col];
        else
            ds_b[ty][tx] = 0.0f;
        
        //barrier to ensure that all the threads complete loading data into tiles
        __syncthreads();

        //calculating the intermediate sum for a specific index
        float c_val = 0.0f;
        for (int i = 0; i<TILE_WIDTH; i++){
            c_val += ds_a[ty][i]*ds_b[i][tx];
        }

        //barrier to ensure that all the threads complete calculation of intermediate results
        __syncthreads();

        //the final result is written to the global memory
        if (row<n && col<k)
            res[row*k+col] = c_val;
    }
}

void mat_mul_launch(float *arr1, float *arr2, float* res, int m, int n, int k){
    int x_dim = 16;
    int y_dim = 16;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // defining the number of threads per block
    dim3 threadsPerBlock(x_dim, y_dim, 1);
    
    // defining the number of blocks per grid
    dim3 blocksPerGrid((k-1)/threadsPerBlock.x+1, (n-1)/threadsPerBlock.y+1, 1);

    cudaEventRecord(start);
    // kernel launch
    mat_mul<<<blocksPerGrid, threadsPerBlock>>>(arr1, arr2, res, m, n, k);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float elapsedTime= 0.0f;
    cudaEventElapsedTime(&elapsedTime,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout<<"gpu time: "<<elapsedTime<<" ms\n";
}
