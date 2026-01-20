#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 4096
// Kernel
__global__ void matrixMulNaive(int *a, int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int temp_sum = 0;
        for (int k = 0; k < n; k++) {
            temp_sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = temp_sum;
    }
}

// Host Code
int main() {
    size_t bytes = N * N * sizeof(int);

    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);

    // Giving values to the matrices
    for(int i = 0; i < N * N; i++) {
        h_a[i] = 1; 
        h_b[i] = 1;
    }
 
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16); 
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Matrix Size: %d x %d\n", N, N);

    cudaEventRecord(start);
    matrixMulNaive<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU Execution Time: %.2f milliseconds\n", milliseconds);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

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
