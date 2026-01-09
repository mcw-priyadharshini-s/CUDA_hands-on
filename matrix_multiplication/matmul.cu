#include <iostream>
#include <cuda_runtime.h>
using namespace std;


#define M 1024
#define K 1024
#define N 1024

void matmul_cpu(float *a,float *b,float *c)
{
    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++)
        {
            float value = 0.0f; 
            for (int k=0;k<K;k++)
            {
                value+=a[i*K+k]*b[k*N+j];
            }
            c[i*N+j]=value;
        }
    }
}

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


int compare_array(float *c_gpu,float *c_cpu)
{
  for(int i=0;i<M;i++)
  {
    for(int j=0;j<N;j++)
    { 
        if (fabs(c_gpu[i*N+j] - c_cpu[i*N+j]) / fabs(c_cpu[i*N+j]) > 1e-5 ) 
        { 
            return false;
        }
        
    }
  }
  return true;
}

int main()
{
    size_t size_a=M*K*sizeof(float);
    size_t size_b=N*K*sizeof(float);
    size_t size_c=M*N*sizeof(float);

    float* h_a=(float*)malloc(size_a);
    float* h_b=(float*)malloc(size_b);
    float* h_c=(float*)malloc(size_c);
    float* h_c_cpu=(float*)malloc(size_c);

    input_array(h_a,h_b);

    float *d_a,*d_b,*d_c;
    
    //-----------------------------------gpu------------------------------------
    
    cudaEvent_t start, stop;
    
    //allocate memory in gpu
    cudaMalloc(&d_a,size_a);
    cudaMalloc(&d_b,size_b);
    cudaMalloc(&d_c,size_c);

    //memcpy from host to device
    cudaMemcpy(d_a,h_a,size_a,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size_b,cudaMemcpyHostToDevice);

    //threads
    dim3 threads(32,32,1);

    //block
    dim3 block(ceil(M/(float)threads.x),ceil(N/(float)threads.y));
    
    //warm_up
    matmul<<<block,threads>>>(d_a,d_b,d_c);
    
    cudaMalloc(&d_c,size_c);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //kernel function
    matmul<<<block,threads>>>(d_a,d_b,d_c);
    
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    
    float elapsedTime= 0.0f;
    cudaEventElapsedTime(&elapsedTime,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_c,d_c,size_c,cudaMemcpyDeviceToHost);

    cout<<"gpu time: "<<elapsedTime<<" ms\n";


    
    //----------------------------------------cpu-------------------------------
    //cpu matmul function
    
    clock_t start_cpu = clock();
    matmul_cpu(h_a,h_b,h_c_cpu);
    clock_t end_cpu = clock();

    // calculate elapse time taken in cpu
    float cpu_time_ms =1000.0*(double)(end_cpu-start_cpu)/CLOCKS_PER_SEC;
    
    cout<<"cpu time: " << cpu_time_ms<<" ms\n";
    

    //results
    int isSame=compare_array(h_c,h_c_cpu);
    if(isSame) cout<< "\nC array calculated in cpu and gpu matches !!!\n";
    else cout<<"\nC array calculated in cpu and gpu doesn't matches :(\n";
    

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);


    return 0;
}