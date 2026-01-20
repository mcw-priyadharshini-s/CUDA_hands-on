#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define MLEN 8

__constant__ int d_mask[MLEN];
//__constant__ int *d_mask = (int*)malloc(MLEN * sizeof(int));

__global__ void conv (int *arr, int *res, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int rad;
    if (MLEN % 2)
        rad = MLEN / 2;
    else
        rad = MLEN/2 - 1;
    int start = tid - rad;
    int temp = 0; 
    if(tid < n){
        for(int i = 0; i<MLEN; i++) {
            if((start + i) >= 0 && (start + i) < n)
                temp += arr[start + i] * d_mask[i];
        }
        res[tid] = temp;
    }  
}

bool verify_res (int *arr, int *mask, int *res, int n) {
    int rad;
    if(MLEN % 2)
        rad = MLEN/2;
    else
        rad = (MLEN/2) - 1;

    int *resin = (int*)malloc(n * sizeof(int));
    for (int i = 0; i<n; i++) {
        int temp = 0;
        int start = i - rad;
        for(int j = 0; j<MLEN; j++) {
            if(start+j >= 0 && start+j < n)
                temp += arr[start+j] * mask[j];
        }
        resin[i] = temp;
        if(resin[i] != res[i]) {
            free(resin);
            return false;
        }   
    }
    free(resin);
    return true;
}

int main(void) {
    int n = 1<<10;
    size_t bytes_n = n*sizeof(int);
    size_t bytes_m = MLEN * sizeof(int);
    int *h_arr, *h_mask, *h_res, *d_arr, *d_res;

    h_arr = (int*)malloc(bytes_n);
    h_mask = (int*)malloc(bytes_m);
    h_res = (int*)malloc(bytes_n);

    cudaMalloc((void**)&d_arr, bytes_n);
    cudaMalloc((void**)&d_res, bytes_n);

    for(int i = 0; i<n; i++) {
        h_arr[i] = rand() % 100;
    }
    for(int i = 0; i<MLEN; i++) {
        h_mask[i] = rand() % 10;
    }

    cudaMemcpyToSymbol(d_mask, h_mask, bytes_m);
    cudaMemcpy(d_arr, h_arr, bytes_n, cudaMemcpyHostToDevice);

    int tpb = 256;
    int bpg = (n+tpb-1) / tpb;

    conv <<<bpg,tpb>>> (d_arr, d_res, n);

    cudaMemcpy(h_res, d_res, bytes_n, cudaMemcpyDeviceToHost);

    if(verify_res(h_arr, h_mask, h_res, n))
        printf("1D convolution completed successfully!!\n");
    else
        printf("1D convolution finished with errors!!\n");

    free(h_arr); free(h_mask); free(h_res);
    cudaFree(d_arr); cudaFree(d_res);
    return 0;
}
