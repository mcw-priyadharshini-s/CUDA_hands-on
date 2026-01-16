#include<stdio.h>
#include<cuda_runtime.h>
#include<cuda/cmath>

__global__ void mat_add(int* mat1, int* mat2, int* mat_res, int rows, int cols) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<rows && j<cols) 
        mat_res[i*cols + j] = mat1[i*cols + j] + mat2[i*cols + j];
}

int main(void) {
    int r, c;
    printf("Enter the number of rows: ");
    scanf("%d", &r);
    printf("\nEnter the number of columns: ");
    scanf("%d", &c);

    int *h_mat1, *h_mat2, *h_mat_res, *d_mat1, *d_mat2, *d_mat_res;
    h_mat1 = (int*)malloc(r*c*sizeof(int));
    h_mat2 = (int*)malloc(r*c*sizeof(int));
    h_mat_res = (int*)malloc(r*c*sizeof(int));
    cudaMalloc((void**)&d_mat1, r*c*sizeof(int));
    cudaMalloc((void**)&d_mat2, r*c*sizeof(int));
    cudaMalloc((void**)&d_mat_res, r*c*sizeof(int));

    for(int i=0; i<r; i++) {
        for(int j=0; j<c; j++) {
            printf("Enter mat1[%d][%d] = ", i, j);
            scanf("%d", &h_mat1[i*c+j]);
        }
    } 
    for(int i=0; i<r; i++) {
        for(int j=0; j<c; j++) {
            printf("Enter mat2[%d][%d] = ", i, j);
            scanf("%d", &h_mat2[i*c+j]);
        }
    } 
    cudaMemcpy(d_mat1, h_mat1, r*c*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, h_mat2, r*c*sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads_per_block(16,16,1);
    dim3 blocks_per_grid(cuda::ceil_div(c,16), cuda::ceil_div(r,16), 1);

    mat_add <<<threads_per_block,blocks_per_grid>>> (d_mat1, d_mat2, d_mat_res, r, c);
    cudaMemcpy(h_mat_res, d_mat_res, r*c*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i<r; i++) {
        for(int j=0; j<c; j++) 
            printf("%d  ",h_mat_res[i*c+j]);
        printf("\n");
    }

    cudaFree(d_mat1); cudaFree(d_mat2); cudaFree(d_mat_res);
    free(h_mat1); free(h_mat2); free(h_mat_res);
    return 0;
}
