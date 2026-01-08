#include "kernel.h"
__global__ void matmulKernel(float *dA, float *dB, float *dC, int m, int n, int k){

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float Cvalue = 0.0f;

    if(row<m && col<k){
        for(int i=0; i<n; i++){
            Cvalue += dA[row*n+i] * dB[i*k+col];
        }
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
