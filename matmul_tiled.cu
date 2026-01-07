#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define tile_size 16

__global__ void matmulTiled(const float *mat1, const float *mat2, float *mat_res, int n) {
    __shared__ float a[tile_size][tile_size], b[tile_size][tile_size];                                                              
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = tile_size * blockIdx.y + ty;
    int col = tile_size * blockIdx.x + tx;

    float sum = 0.0f;

    int num_tiles = (n + tile_size - 1) / tile_size;

    for(int t = 0; t < num_tiles; t++) {
        int tiled_row = tile_size * t + ty;
        int tiled_col = tile_size * t + tx;

        if(row < n && tiled_col < n)
            a[ty][tx] = mat1[row * n + tiled_col];
        else
            a[ty][tx] = 0.0f;

        if(col < n && tiled_row < n)
            b[ty][tx] = mat2[tiled_row * n + col]; 
        else
            b[ty][tx] = 0.0f;

        __syncthreads();

        for(int k = 0; k < tile_size; k++) 
            sum += (a[ty][k] * b[k][tx]);
        
        __syncthreads();
    }
    if(row < n && col < n) 
        mat_res[row * n + col] = sum;
}

float* matmulCpu(const float *mat1, const float *mat2, int n) {
    float *matres = (float*)malloc(n*n*sizeof(int));
    for(int i = 0; i<n; i++) {
        for(int j = 0; j<n; j++) {
            float sum = 0.0f;
            for(int k = 0; k<n; k++) 
                sum += (mat1[i * n + k] * mat2[k *n + j]);
            matres[i * n + j] = sum;
        }
    }
    return matres;
}

bool matmulComparison (const float *mat1, const float *mat2, int n, float epsilon = 0.0001) {
    for(int i = 0; i<n; i++) {
        for(int j = 0; j<n; j++) {
            if(fabs(mat1[i*n+j] - mat2[i*n+j]) > epsilon) 
                return false;
        }
    }
    return true;
}

int main() {
    // assign dim of matrices
    int n = 1024;

    // declare host and device matrices 
    float *h_mat1, *h_mat2, *h_mat_res, *h_mat_cpu, *d_mat1, *d_mat2, *d_mat_res;

    h_mat1 = (float*)malloc(n*n*sizeof(float));
    h_mat2 = (float*)malloc(n*n*sizeof(float));
    h_mat_res = (float*)malloc(n*n*sizeof(float));
    h_mat_cpu = (float*)malloc(n*n*sizeof(float));

    cudaMalloc((void**)&d_mat1, n*n*sizeof(float));
    cudaMalloc((void**)&d_mat2, n*n*sizeof(float));
    cudaMalloc((void**)&d_mat_res, n*n*sizeof(float));

    // assign values to host mat
    for(int i = 0; i<n; i++) {
        for(int j = 0; j<n; j++) {
            h_mat1[i*n+j] = rand() % 10;
            h_mat2[i*n+j] = rand() % 20;
        }
    }

    // copy to device
    cudaMemcpy(d_mat1, h_mat1, n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, h_mat2, n*n*sizeof(float), cudaMemcpyHostToDevice);

    // call kernel
    dim3 tpb(16,16,1); //taking threads in block same as number of threads per tile (tile_size)
    dim3 bpg((n+tile_size-1)/tile_size, (n+tile_size-1)/tile_size, 1);
    matmulTiled <<<bpg,tpb>>> (d_mat1, d_mat2, d_mat_res, n);

    // copy result to host_res
    cudaMemcpy(h_mat_res, d_mat_res, n*n*sizeof(float), cudaMemcpyDeviceToHost);

    // pass inputs to cpu matmul function
    h_mat_cpu = matmulCpu(h_mat1, h_mat2, n);

    // pass gpu and cpu results to comparison function
    if(matmulComparison(h_mat_cpu, h_mat_res, n))
        printf("\n\tSuccess!! :)\n\n");
    else
        printf("\n\tDistinct!! :(\n\n");

    // free and cudafree all matrices
    free(h_mat1); free(h_mat2); free(h_mat_res); free(h_mat_cpu);
    cudaFree(d_mat1); cudaFree(d_mat2); cudaFree(d_mat_res);

    return 0;
}