#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024 
#define BLOCK_DIM 16 
#define TILE_DIM (BLOCK_DIM * 2)

__global__ void matrixMulRegisterTiled(int *a, int *b, int *c, int n) {
    // Tile
    __shared__ int tile_a[TILE_DIM][TILE_DIM];
    __shared__ int tile_b[TILE_DIM][TILE_DIM];

    // Register
    int c_reg[2][2] = {0, 0, 0, 0};

    int tx = threadIdx.x; 
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_DIM + ty * 2;
    int col = blockIdx.x * TILE_DIM + tx * 2;

    // Loop over tiles
    for (int m = 0; m < n; m += TILE_DIM) {

        int tid = ty * BLOCK_DIM + tx;
        for (int i = 0; i < 4; i++) {
            int tile_idx = tid + i * (BLOCK_DIM * BLOCK_DIM); 
            int r = tile_idx / TILE_DIM;
            int c_loc = tile_idx % TILE_DIM;
            
            tile_a[r][c_loc] = a[(blockIdx.y * TILE_DIM + r) * n + (m + c_loc)];
            tile_b[r][c_loc] = b[(m + r) * n + (blockIdx.x * TILE_DIM + c_loc)];
        }
        
        __syncthreads(); 

        for (int k = 0; k < TILE_DIM; k++) {
            int a_val0 = tile_a[ty * 2][k];
            int a_val1 = tile_a[ty * 2 + 1][k];
            int b_val0 = tile_b[k][tx * 2];
            int b_val1 = tile_b[k][tx * 2 + 1];

            c_reg[0][0] += a_val0 * b_val0;
            c_reg[0][1] += a_val0 * b_val1;
            c_reg[1][0] += a_val1 * b_val0;
            c_reg[1][1] += a_val1 * b_val1;
        }

        __syncthreads();
    }

    c[(row) * n + col]         = c_reg[0][0];
    c[(row) * n + (col + 1)]   = c_reg[0][1];
    c[(row + 1) * n + col]     = c_reg[1][0];
    c[(row + 1) * n + (col + 1)] = c_reg[1][1];
}

int main() {
    size_t bytes = N * N * sizeof(int);
    
    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);
    
    for(int i = 0; i < N*N; i++) { h_a[i] = 1; h_b[i] = 1; }

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks(N / TILE_DIM, N / TILE_DIM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMulRegisterTiled<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Register Tiled Time: %.3f ms\n", milliseconds);

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