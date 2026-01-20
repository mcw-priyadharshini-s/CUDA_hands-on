#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MLEN 75
__constant__ int d_mask[MLEN];

__global__ void conv (int *arr, int *res, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int rad = MLEN/2;

    extern __shared__ int sh[];

    int offset = blockDim.x + threadIdx.x;
    int g_offset = blockDim.x * blockIdx.x + offset;
    int n_p = blockDim.x + 2*rad;

    sh[threadIdx.x] = arr[tid];
    if (offset < n_p) {
        sh[offset] = arr[g_offset];
    }
    __syncthreads();

    int temp = 0;
    for(int j = 0; j<MLEN; j++) {
        temp += sh[threadIdx.x + j] * d_mask[j];
    }
    if(tid < n) {
        res[tid] = temp;
    }
}

bool verify_res (int *arr, int *mask, int *res, int n) {
    int temp;
    for(int i = 0; i<n; i++) {
        temp = 0;
        for(int j = 0; j<MLEN; j++) {
            temp += arr[i+j] * mask[j];
        }
        if(temp - res[i]) {
            return false;
        } 
    }
    return true;
}

int main() {
    int n = 1<<10;

    int rad;
    rad = MLEN/2;
    int n_p = n + 2 * rad;

    int *h_arr, *h_mask, *h_res, *d_arr, *d_res;
    h_arr = (int*)malloc(n_p * sizeof(int));
    h_mask = (int*)malloc(MLEN * sizeof(int));
    h_res = (int*)malloc(n * sizeof(int));
    cudaMalloc((void**)&d_arr, n_p * sizeof(int));
    cudaMalloc((void**)&d_res, n * sizeof(int));

    for(int i = 0; i<n_p; i++) {
        if((i < rad) || (i >= n+rad)) {
            h_arr[i] = 0;
        }
        else {
            h_arr[i] = rand() % 100;
        } 
    }
    for(int j = 0; j<MLEN; j++) {
        h_mask[j] = rand() % 10;
    } 

    cudaMemcpyToSymbol(d_mask, h_mask, MLEN * sizeof(int));
    cudaMemcpy(d_arr, h_arr, n_p * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    size_t shmem = (threads + 2*rad) * sizeof(int);

    conv <<<blocks,threads,shmem>>> (d_arr, d_res, n);

    cudaMemcpy(h_res, d_res, n * sizeof(int), cudaMemcpyDeviceToHost);

    if(verify_res(h_arr, h_mask, h_res, n)) {
        printf("Tiled 1D Convolution successful!! :)\n");
    }
    else {
        printf("Tiled 1D convolution finished with errors!! :(\n");
    }

    free(h_arr); free(h_mask); free(h_res);
    cudaFree(d_arr); cudaFree(d_res);
    return 0;
}