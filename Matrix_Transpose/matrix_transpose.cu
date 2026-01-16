#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

__global__ void transpose(const int *mat, int *mat_res, int n) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<n && j<n)  {
        mat_res[j*n + i] = mat[i*n + j];
    }   
} 

int main(void) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    int n = 1024;
    //int rows, cols;
    //printf("Enter number of rows : ");
    //scanf("%d", &rows);
    //printf("\nEnter number of columns : ");
    //scanf("%d", &cols);

    int* h_mat = (int*)malloc(n*n*sizeof(int));
    int* d_mat; cudaMalloc((void**)&d_mat, n*n*sizeof(int));
    int* d_mat_res; cudaMalloc((void**)&d_mat_res, n*n*sizeof(int));

    //for(int i = 0; i<rows; i++) {
    //    for(int j = 0; j<cols; j++) {
    //        printf("\nEnter element mat[%d][%d] = ", i, j);
    //        scanf("%d", &h_mat[i*cols + j]);
    //    }
    //}

    for(int i = 0; i<n; i++) {
        for(int j = 0; j<n; j++) 
            h_mat[i * n + j] = rand() % 50;
    }

    cudaMemcpy(d_mat, h_mat, n*n*sizeof(int), cudaMemcpyHostToDevice);

    int numcol_blocks = (n + 16 -1) / 16, numrow_blocks = (n + 16 - 1) / 16;
    dim3 threads(16,16,1); 
    dim3 blocks(numcol_blocks, numrow_blocks, 1);

    cudaEventRecord(start);
    transpose <<<blocks,threads>>> (d_mat, d_mat_res, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float millisec = 0;
    cudaEventElapsedTime(&millisec, start, stop);

    cudaMemcpy(h_mat, d_mat_res, n*n*sizeof(int), cudaMemcpyDeviceToHost);

    //printf("\nResult matrix :\n");
    //for(int i = 0; i<n; i++) {
    //    for(int j = 0; j<n; j++) 
    //        printf("%d\t", h_mat[i*n+j]);
    //    printf("\n");
    //}
    cudaFree(d_mat); cudaFree(d_mat_res);
    free(h_mat);

    printf("\nKernel Execution Time : %0.3f ms", millisec);
    return 0;
}
