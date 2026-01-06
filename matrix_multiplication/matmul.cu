#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
using namespace std;


#define M 400
#define K 300
#define N 500

void matmul_cpu(float *a,float *b,float *c)
{
    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++)
        {
            for (int k=0;k<K;k++)
            {
                c[i*N+j]+=a[i*K+k]*b[k*N+j];
            }
        }
    }
}

__global__ void matmul(float *a,float *b,float *c)
{
   int row=(blockDim.x*blockIdx.x)+threadIdx.x;
   int col=(blockDim.y*blockIdx.y)+threadIdx.y;

   if(row<M && col<N)
   {
    for(int k=0;k<K;k++)
    {
        c[row*N+col]+=a[row*K+k]*b[k*N+col];
    }
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

void print_array(float *a,float *b,float *c)
{
    cout<<"array a\n";
    for(int i=0;i<M*K;i++)
    {
        if (i!=0 && i%K==0)
        {
            cout<<"\n"<<setw(6)<<a[i]<<" ";
        }
        else
        {
            cout<<setw(6)<<a[i]<<" ";
        }
    }
    cout<<"\narray b\n";
    for(int i=0;i<K*N;i++)
    {
        if (i!=0 && i%N==0)
        {
            cout<<"\n"<<setw(6)<<b[i]<<" ";
        }
        else
        {
            cout<<setw(6)<<b[i]<<" ";
        }
    }
    cout<<"\narray c\n";
    for(int i=0;i<M*N;i++)
    {
        if (i!=0 && i%N==0)
        {
            cout<<"\n"<<setw(6)<<c[i]<<" ";
        }
        else
        {
            cout<<setw(6)<<c[i]<<" ";
        }
    }
    cout<<"\n";
}

int main()
{
    size_t size_a=M*K*sizeof(float);
    size_t size_b=N*K*sizeof(float);
    size_t size_c=M*N*sizeof(float);

    float* h_a=(float*)malloc(size_a);
    float* h_b=(float*)malloc(size_b);
    float* h_c=(float*)malloc(size_c);

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
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        cout<<"CUDA error: "<< cudaGetErrorString(err)<<"\n";
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    
    float elapsedTime= 0.0f;
    cudaEventElapsedTime(&elapsedTime,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    

    

    //----------------------------------------cpu-------------------------------
    //cpu matmul function
    
    clock_t start_cpu = clock();
    matmul_cpu(h_a,h_b,h_c);
    clock_t end_cpu = clock();

    // calculate elapse time taken in cpu
    float cpu_time_ms =1000.0*(double)(end_cpu-start_cpu)/CLOCKS_PER_SEC;


    //results
    
    cout<<"----------------------------------------cpu-------------------------------\n";
    // print_array(h_a,h_b,h_c);
    cout<<"cpu time: " << cpu_time_ms<<" ms\n";

    cout<<"----------------------------------------gpu-------------------------------\n";
    cudaMemcpy(h_c,d_c,size_c,cudaMemcpyDeviceToHost);
    // print_array(h_a,h_b,h_c);
    cout<<"gpu time: "<<elapsedTime<<" ms\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);


    return 0;
}