#include "kernel.h"

int main(){

    //Getting dimensions for matrices from the user
    int n;

    cout<<"Enter number of elements: ";
    cin>>n;

    //Declaration of host arrays
    float *hA = new float[n];
    float *hB = new float[n];
    float *hC = new float[n];

    //Declaration of device arrays
    float *dA, *dB, *dC;

    size_t size = n*sizeof(float);


    //Device Memory allocation 
    cudaMalloc((void **)&dA, size);
    cudaMalloc((void **)&dB, size);
    cudaMalloc((void **)&dC, size);

    //Filling values for the matrices
    for(int i=0; i<n; i++){
        hA[i] = i+1;
        hB[i] = i+1;
    }

    //Copying values from host to device
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    //Calling the function for kernel launch
    launchKernel(dA, dB, dC, n);

    //Copying results from device to host
    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

    //Print resultant array
    for(int i=0; i<n; i++){
        cout<<hC[i]<<" ";
    }

    //Release resources
    cudaFree(dA); 
    cudaFree(dB);
    cudaFree(dC);

    delete[] hA;
    delete[] hB;
    delete[] hC;

    cudaDeviceSynchronize();

    return 0;
}

