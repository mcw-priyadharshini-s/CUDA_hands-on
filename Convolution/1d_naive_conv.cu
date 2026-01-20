#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void conv (int *vec1, int *vec2, int *vecres, int n, int m) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int temp = 0;

    int rad;
    if(m % 2)
        rad = m/2;
    else
        rad = (m/2) - 1;
    int start = tid - rad;
    if(tid < n) {
        for(int j = 0; j<m; j++) {
            if((start + j) >= 0 && (start + j) < n)
                temp += vec1[start+j] * vec2[j];
        }
        vecres[tid] = temp;
    }
}

bool verify_res (int *vec1, int *vec2, int *vecres, int n, int m) {
    int *res = (int*)malloc(n * sizeof(int));
    int rad;
    if(m % 2) //checking if number of mask elements are even or odd
        rad = m/2;
    else
        rad = (m/2) - 1;

    for (int i = 0; i<n; i++) {
        int start = i - rad;
        int temp = 0;
        for(int j = 0; j<m; j++) {
            if((start+j) >= 0 && (start+j) < n)
                temp += vec1[start + j] * vec2[j];
        }
        res[i] = temp;
        if(res[i] - vecres[i])
            return false;
    }
    //if(res == vecres) //not correct sinc this compares the memory locations and not the value of array
    //    return true;
    //return false;
    return true;
} 

int main(void) {
    int n = 1 << 10;
    size_t bytes = n * sizeof(int);
    int m = 7;

    int *h_vec1, *h_vec2, *h_vecres;
    int *d_vec1, *d_vec2, *d_vecres;

    h_vec1 = (int*)malloc(bytes);
    h_vec2 = (int*)malloc(m * sizeof(int));
    h_vecres = (int*)malloc(bytes);

    cudaMalloc((void**)&d_vec1, bytes);
    cudaMalloc((void**)&d_vec2, m * sizeof(int));
    cudaMalloc((void**)&d_vecres, bytes);

    for(int i = 0; i<n; i++) {
        h_vec1[i] = rand() % 100;
    }
    for(int i = 0; i<m; i++) {
        h_vec2[i] = rand() % 10;
    }

    cudaMemcpy(d_vec1, h_vec1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, h_vec2, m * sizeof(int), cudaMemcpyHostToDevice);

    int threads_per_blk = 256;
    int blk_per_grid = (n + threads_per_blk - 1) / threads_per_blk;

    conv <<<blk_per_grid,threads_per_blk>>> (d_vec1, d_vec2, d_vecres, n, m);

    cudaMemcpy(h_vecres, d_vecres, bytes, cudaMemcpyDeviceToHost);

    if(verify_res(h_vec1, h_vec2, h_vecres, n, m)) 
        printf("1D convolution completed successfully!\n");
    else
        printf("1D convolution finished with errors!!\n");

    free(h_vec1); free(h_vec2); free(h_vecres);
    cudaFree(d_vec1); cudaFree(d_vec2); cudaFree(d_vecres);
    return 0;
}
