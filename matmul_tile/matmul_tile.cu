#include <iostream>
#include <cuda_runtime.h>
using namespace std;


#define M 1024
#define K 1024
#define N 1024

#define TILE_WIDTH 64

__global__ void matmul(float *a,float *b,float *c)
{
   int col=(blockDim.x*blockIdx.x)+threadIdx.x;
   int row=(blockDim.y*blockIdx.y)+threadIdx.y;

   if(row<M && col<N)
   {
    float value = 0.0f; 

    for(int k=0;k<K;k++) value+=a[row*K+k]*b[k*N+col];

    c[row * N + col] = value;

   }
}

__global__ void matmul_tile(float *a,float *b,float *c)
{
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    int col=blockIdx.x*TILE_WIDTH+threadIdx.x;
    int row=blockIdx.y*TILE_WIDTH+threadIdx.y;

    float value=0.0f;
    int A_col,B_row;
    
    for(int t=0;t<(K+TILE_WIDTH-1)/TILE_WIDTH;t++)
    { 
        
        A_col=t*TILE_WIDTH+threadIdx.x;
        B_row=t*TILE_WIDTH+threadIdx.y;
        

        if(row<M && A_col<K)
        {
            sh_A[threadIdx.y][threadIdx.x]=a[row*K+A_col];
        }
        else
        {
            sh_A[threadIdx.y][threadIdx.x]=0.0f;
        }
        if(B_row<K && col<N)
        {
            sh_B[threadIdx.y][threadIdx.x]=b[B_row*N+col];
        }
        else
        {
            sh_B[threadIdx.y][threadIdx.x]=0.0f;
        }
        __syncthreads();
        
        for (int i = 0; i < TILE_WIDTH; i++)
            value += sh_A[threadIdx.y][i] * sh_B[i][threadIdx.x];

        __syncthreads();

    }

    if (row < M && col < N)
    {
        c[row * N + col] = value;
    }

}

int compare_array(float *c_naive,float *c_tiled)
{
  for(int i=0;i<M;i++)
  {
    for(int j=0;j<N;j++)
    { 
        if (fabs(c_tiled[i*N+j] - c_naive[i*N+j]) / fabs(c_naive[i*N+j]) > 1e-5 ) 
           return false;
    }
  }
  return true;
}

void input_array(float *a,float *b)
{
    int max_val=M*K;
    for(int i=0;i<M*K;i++)
    {
        a[i]=(float)(rand()%(max_val+1)+1);
    }
    max_val=K*N;
    for(int i=0;i<K*N;i++)
    {
        b[i]=(float)(rand()%(max_val+1)+1);
    }
}

void matmul_tiled(float *d_a,float *d_b,float *d_c,size_t size_c, float *h_c_tiled)
{
    cudaEvent_t start, stop;

    //threads
    dim3 threads(TILE_WIDTH,TILE_WIDTH,1);
    
    //block
    dim3 block((N + TILE_WIDTH - 1) / TILE_WIDTH,
            (M + TILE_WIDTH - 1) / TILE_WIDTH);


   //warm_up
    matmul_tile<<<block,threads>>>(d_a,d_b,d_c);
    

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul_tile<<<block,threads>>>(d_a,d_b,d_c);
    
    
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    
    float elapsedTime= 0.0f;

    cudaEventElapsedTime(&elapsedTime,start,stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cout<<"----------------------------------------gpu tiled-------------------------------\n";
    cudaMemcpy(h_c_tiled,d_c,size_c,cudaMemcpyDeviceToHost);
    cout<<" gpu time: "<<elapsedTime<<" ms\n";
}

void matmul_naive(float *d_a,float *d_b, float *d_c,size_t size_c,float *h_c_naive)
{
    cudaEvent_t start, stop;

    //thread
    dim3 threads(32,32,1);
    
    //block
    dim3 block(ceil(M/(float)threads.x),ceil(N/(float)threads.y));

    //warm_up
    matmul<<<block,threads>>>(d_a,d_b,d_c);
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul<<<block,threads>>>(d_a,d_b,d_c);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    
    float elapsedTime= 0.0f;

    cudaEventElapsedTime(&elapsedTime,start,stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout<<"----------------------------------------gpu naive-------------------------------\n";
    cudaMemcpy(h_c_naive,d_c,size_c,cudaMemcpyDeviceToHost);
    cout<<" gpu time: "<<elapsedTime<<" ms\n";
      
}
int main()
{
    size_t size_a=M*K*sizeof(float);
    size_t size_b=N*K*sizeof(float);
    size_t size_c=M*N*sizeof(float);

    float* h_a=(float*)malloc(size_a);
    float* h_b=(float*)malloc(size_b);
    float* h_c_naive=(float*)malloc(size_c);
    float* h_c_tiled=(float*)malloc(size_c);

    input_array(h_a,h_b);

    float *d_a,*d_b,*d_c;
    
    
    //allocate memory in gpu
    cudaMalloc(&d_a,size_a);
    cudaMalloc(&d_b,size_b);
    cudaMalloc(&d_c,size_c);


    //memcpy from host to device
    cudaMemcpy(d_a,h_a,size_a,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size_b,cudaMemcpyHostToDevice);
    
    matmul_naive(d_a,d_b,d_c,size_c,h_c_naive);
    
    matmul_tiled(d_a,d_b,d_c,size_c,h_c_tiled);
    
    //result
    int isSame=compare_array(h_c_naive,h_c_tiled);
    if(isSame) cout<< "\nC array calculated with tiling and without tiling matches !!!\n";
    else cout<<"\nC array calculated with tiling and without tiling doesn't matches :(\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c_naive);
    free(h_c_tiled);

    return 0;
}