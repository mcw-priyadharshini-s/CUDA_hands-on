#include<cuda_runtime.h>
#include<stdio.h>

__global__ void vec_add(double* a, double* b, double *c, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N) 
        c[idx] = a[idx] + b[idx];
}

int main(void) {
    int N;
    printf("enter number of elements: ");
    scanf("%d", &N);

    double *h_a = (double*)malloc(N*sizeof(double));
    double *h_b = (double*)malloc(N*sizeof(double));
    double *h_c = (double*)malloc(N*sizeof(double));

    for(int i = 0; i < N; i++) {
        h_a[i] = 1.00;
        h_b[i] = 2.00;
    }

    double *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, N*sizeof(double));
    cudaMalloc((void**)&d_b, N*sizeof(double));
    cudaMalloc((void**)&d_c, N*sizeof(double));

    cudaMemcpy(d_a, h_a, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(double), cudaMemcpyHostToDevice);

    //dim3 grid_size(2,2,0); dim3 block_size(4,3,0);

    int threads_per_block = 256;
    int blocks_per_grid = ceil(float(N)/threads_per_block);

    vec_add <<< threads_per_block, blocks_per_grid >>> (d_a, d_b, d_c, N);
    
    cudaMemcpy(h_c, d_c, N*sizeof(double), cudaMemcpyDeviceToHost);

    printf("Vectors added!!\n");
    for(int i = 0; i < N; i++) 
        printf("%lf\t", h_c[i]);
    
    free(h_c); free(h_b); free(h_a);
    cudaFree(d_c); cudaFree(d_b); cudaFree(d_a);
    return 0;    
}
