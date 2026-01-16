#include <stdio.h>
#include <cuda_runtime.h>

#define tile_size 16

__global__ void transTiled (const float *matip, float *matres, int n) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int ip_row = blockIdx.y * tile_size + ty;
    int ip_col = blockIdx.x * tile_size + tx;
    __shared__ float a[tile_size][tile_size];
    if(ip_row < n && ip_col < n)
        a[ty][tx] = matip[ip_row * n + ip_col];
    else
        a[ty][tx] = 0.0f;
    __syncthreads(); 
    // indexing to find where to put tile in output matrix
    int op_row = blockIdx.x * tile_size + ty;
    int op_col = blockIdx.y * tile_size + tx;
    if(op_row < n && op_col < n)
        matres[op_row * n + op_col] = a[tx][ty]; // transposes the element in the tile
}

float* transCpu (const float *mat, int n) {
    float* result = (float*)malloc(n*n*sizeof(float));
    for(int i = 0; i<n; i++) {
        for(int j = 0; j<n; j++) 
            result[j * n + i] = mat[i * n + j];
    }
    return result;
} 

bool transComp (const float *mat1, const float *mat2, int n, float epsilon = 0.0001) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(fabs(mat1[i*n+j] - mat2[i*n+j]) > epsilon)
                return false;
        }
    }
    return true;
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop); 
    int n = 1024;
    float *h_mat1, *h_matres, *d_mat1, *d_matres;
    size_t bytes = n * n * sizeof(float);
    h_mat1 = (float*)malloc(bytes);
    h_matres = (float*)malloc(bytes);
    cudaMalloc((void**)&d_mat1, bytes);
    cudaMalloc((void**)&d_matres, bytes);

    for(int i = 0; i<n; i++) {
        for(int j = 0; j<n; j++) 
            h_mat1[i * n + j] = rand() % 50;
    }

    cudaMemcpy(d_mat1, h_mat1, bytes, cudaMemcpyHostToDevice);

    dim3 tpb(tile_size, tile_size, 1);
    int num_blocks = (n + tile_size - 1) / tile_size;
    dim3 bpg(num_blocks, num_blocks, 1);

    cudaEventRecord(start);
    transTiled <<<bpg,tpb>>> (d_mat1, d_matres, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_matres, d_matres, bytes, cudaMemcpyDeviceToHost);

    float millisec = 0;
    cudaEventElapsedTime(&millisec, start, stop);

    float* trans_cpu = (float*)malloc(bytes);
    trans_cpu = transCpu(h_mat1, n);

    if(transComp(trans_cpu, h_matres, n))
        printf("Transposes found equal!! :)\n");
    else
        printf("Transposes found distinct!! :(\n");

    free(h_mat1); free(trans_cpu);
    cudaFree(d_mat1); cudaFree(d_matres);

    printf("\nKernel Execution Time : %0.3f ms", millisec);
    return 0;
}
