#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

const int N= 500000000;

__global__ void vectorAdd(float *a,float *b,float *c)
{
   int i=(blockDim.x*blockIdx.x)+threadIdx.x;
   if (i<N){

       c[i]=a[i]+b[i];
   }
}

void vect_add_cpu(float *a,float *b,float *c)
{
    for (int i=0;i<N;i++)
    {
        c[i]=a[i]+b[i];
    }
}

void random_input(float *arr)
{
    for (int i=0;i<N;i++)
    {
        arr[i]=(float)rand()/RAND_MAX;
    }
}

void print_result(float *arr)
{
    for(int i=0;i<N;i++)
    {
        printf("%f ",arr[i]);
    }
    printf("\n");
}

void gpu_details()
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;

    cudaGetDeviceProperties(&props,device);

    printf("device name: %s\n",props.name);
    printf("Max thread per block: %d\n",props.maxThreadsPerBlock);
    printf("Max threads per multiprocessors: %d\n",props.maxThreadsPerMultiProcessor);
    printf("Max block dimensions: x=%d, y=%d, z=%d\n",props.maxThreadsDim[0],props.maxThreadsDim[1],props.maxThreadsDim[2]);
    printf("Max grid dimensions: x=%d, y=%d, z=%d\n",
    props.maxGridSize[0],props.maxGridSize[1],props.maxGridSize[2]);
    printf("Number of multiprocessors: %d\n\n",props.multiProcessorCount);
}


int main()
{
    
    size_t size = N*sizeof(float);
    cudaEvent_t start, stop;
    float elapsedTime;
    float *A,*B,*C;
    float *a_ar = (float*)malloc(N * sizeof(float));
    float *b_ar = (float*)malloc(N * sizeof(float));
    float *c_ar = (float*)malloc(N * sizeof(float));
    dim3 threads(1024,1,1);
    dim3 blocks(ceil(N/(float)threads.x));
    double cpu_time_ms;

    random_input(a_ar);
    random_input(b_ar);

    // -----------------------------------------------------------------------------
    printf("\n-----------------------vector addition------------------\n\n");
    printf("GPU details\n\n");
    gpu_details();
    printf("Config set for this vector addition kernel:\n\n");

    printf("total elements: %d\n\n",N);

    printf("GPU configs set by user\n\n");
    printf("threads: %d %d %d\n",threads.x,threads.y,threads.z);
    printf("blocks %d %d %d\n",blocks.x,blocks.y,blocks.z);
    printf("---------------------------------------------------------------\n");

    

    //-----------------------gpu parallel compute-------------------------------------
    
    //create event
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //memory allocation in gpu
    cudaMalloc(&A,size);
    cudaMalloc(&B,size);
    cudaMalloc(&C,size);

    //memory copy from host (cpu) to device (gpu)
    cudaMemcpy(A,a_ar,size,cudaMemcpyHostToDevice);
    cudaMemcpy(B,b_ar,size,cudaMemcpyHostToDevice);
    

    //warm up
    vectorAdd<<<blocks,threads>>>(A,B,C);
    
   //record start time
    cudaEventRecord(start, 0);

    // parallel compute kernel function
    vectorAdd<<<blocks,threads>>>(A,B,C);
    
    //record end time
    cudaEventRecord(stop,0);

    //wait till gpu completes
    cudaEventSynchronize(stop);

    //calculate elapse time
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("\ngpu time: %f ms\n",elapsedTime);

    //destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //mem cpy result from device (gpu) to host (cpu)
    cudaMemcpy(c_ar,C,size,cudaMemcpyDeviceToHost);
    
    // print results
    // print_result(c_ar,N);

    //free memory in device (gpu)
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    //------------------------------------cpu ----------------------------------
    
    // record start time
    clock_t start_cpu = clock();

    // run function in cpu
    vect_add_cpu(a_ar,b_ar,c_ar);

    // record end time
    clock_t end_cpu = clock();

    // calculate elapse time taken in cpu
    cpu_time_ms =1000.0 *(double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

    printf("cpu time: %f ms\n\n",cpu_time_ms);

    // print results
    // print_result(c_ar,N);
    
    return 0;
}