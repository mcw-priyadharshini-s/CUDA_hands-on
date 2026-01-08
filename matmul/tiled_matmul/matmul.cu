#include "kernel.h"
__global__ void matmulKernel(float *dA, float *dB, float *dC, int m, int n, int k){

    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float Cvalue = 0.0f;

    //Phase loop
    for(int t=0; t<(n-1)/TILE_WIDTH+1; t++){
        //Loading elements into shared memory
        if(row<m && (t*TILE_WIDTH+tx)<n)
            ds_A[ty][tx] = dA[row*n + (t*TILE_WIDTH+tx)];
        else
            ds_A[ty][tx] = 0.0f;
        if(t*TILE_WIDTH+ty<n && col<k)
            ds_B[ty][tx] = dB[(t*TILE_WIDTH+ty)*k + col];
        else    
            ds_B[ty][tx] = 0.0f;

        __syncthreads();

        //Computation of dot product
        for(int i=0; i<TILE_WIDTH; i++){
            Cvalue += ds_A[ty][i] * ds_B[i][tx];
        }
        __syncthreads();
    }
    //Loading final result
    if(row<m && col<k){
        dC[row*k+col] = Cvalue;        
    }
}

void launchKernel(float *dA, float *dB, float *dC, int m, int n, int k){
    
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
    matmulKernel<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, m, n, k);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float elapsedTime= 0.0f;
    cudaEventElapsedTime(&elapsedTime,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout<<"GPU execution time: "<<elapsedTime<<" ms\n";
}
