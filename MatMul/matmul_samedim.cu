#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda/cmath>

__global__ void matmul (const int *mat1, const int *mat2, int *matop, int n) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<n && j<n) {
        int sum = 0;
        for(int k = 0; k<n; k++) 
            sum += (mat1[i*n + k] * mat2[k*n + j]);
        matop[i*n + j] = sum;
    } 
}

int *matmul_cpu(const int *mat1, const int *mat2, int n) {
    int *matop = (int*)malloc(n*n*sizeof(int));
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            int sum = 0;
            for(int k=0; k<n; k++) {
                sum += (mat1[i*n + k] * mat2[k*n + j]);
            }
            matop[i*n + j] = sum;
        }
    }
    return matop;
}

bool matmul_comp(const int *mat1, const int *mat2, int n, float epsilon=0.000001) {
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            if(fabs(mat1[i*n+j] - mat2[i*n+j]) > epsilon) 
                return false;
        }
    }
    return true;
}

int main() {
    int n = 3;
    int *h_mat1, *h_mat2, *h_mat_res, *d_mat1, *d_mat2, *d_mat_res;
    h_mat1 = (int*)malloc(n*n*sizeof(int));
    h_mat2 = (int*)malloc(n*n*sizeof(int));
    h_mat_res = (int*)malloc(n*n*sizeof(int));

    cudaMalloc((void**)&d_mat1, n*n*sizeof(int));
    cudaMalloc((void**)&d_mat2, n*n*sizeof(int));
    cudaMalloc((void**)&d_mat_res, n*n*sizeof(int));

    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            printf("\nEnter mat1[%d][%d] = ", i, j);
            scanf("%d", &h_mat1[i*n + j]);
        }
    }
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            printf("\nEnter mat2[%d][%d] = ", i, j);
            scanf("%d", &h_mat2[i*n + j]);
        }
    }

    cudaMemcpy(d_mat1, h_mat1, n*n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, h_mat2, n*n*sizeof(int), cudaMemcpyHostToDevice);

    dim3 tpb(16,16,1);
    dim3 bpg(cuda::ceil_div(n,16), cuda::ceil_div(n,16), 1); //of result

    matmul <<<tpb,bpg>>> (d_mat1, d_mat2, d_mat_res, n);

    cudaMemcpy(h_mat_res, d_mat_res, n*n*sizeof(int), cudaMemcpyDeviceToHost);

    //cudaDeviceSynchronize();

    if(matmul_comp(h_mat_res, matmul_cpu(h_mat1, h_mat2, n), n))
        printf("\nCongratulations! Vectors are equal..!!!");
    else 
        printf("\nVectors are distinct! :(");

    printf("\nMatrix on GPU : \n");
    for(int i=0; i<n; i++) {
        printf("\n");
        for(int j=0; j<n; j++) 
            printf("%d\t", h_mat_res[i*n+j]);
    }

    int* mat_cpu = (int*)malloc(n*n*sizeof(int));
    mat_cpu = matmul_cpu(h_mat1, h_mat2, n);
    printf("\nMatrix on CPU : \n");
    for(int i=0; i<n; i++) {
        printf("\n");
        for(int j=0; j<n; j++) 
            printf("%d\t", mat_cpu[i*n+j]);
    }

    free(h_mat1); free(h_mat2); free(h_mat_res);
    cudaFree(d_mat1); cudaFree(d_mat2); cudaFree(d_mat_res);
    return 0;
}
