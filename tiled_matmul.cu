#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 4096 
#define TILE_SIZE 16

// Kernel
__global__ void matrixMulTiled(int *a, int *b, int *c, int n) {
    __shared__ int tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ int tile_b[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    int temp_sum = 0;

    for (int m = 0; m < n; m += TILE_SIZE) {

        tile_a[ty][tx] = a[row * n + (m + tx)];
        tile_b[ty][tx] = b[(m + ty) * n + col];

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            temp_sum += tile_a[ty][k] * tile_b[k][tx];
        }

        __syncthreads();
    }

    c[row * n + col] = temp_sum;
}

// Host Code
int main() {
    size_t bytes = N * N * sizeof(int);
    
    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);
    
    // Giving Values to Matrices
    for(int i = 0; i < N*N; i++) { 
        h_a[i] = 1; 
        h_b[i] = 1; 
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes); 
    cudaMalloc(&d_b, bytes); 
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    

    printf("Matrix Size: %d x %d\n", N, N);

    cudaEventRecord(start);
    matrixMulTiled<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Tiled Time: %.3f ms\n", milliseconds);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    printf("Verification (0,0): %d \n", h_c[0]);

    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_c);
    free(h_a); 
    free(h_b); 
    free(h_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}