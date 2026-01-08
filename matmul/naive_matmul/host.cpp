#include "kernel.h"

int main(){

    //Getting dimensions for matrices from the user
    int m, n, k;

    cout<<"Enter number of rows for matrix1: ";
    cin>>m;

    cout<<"Enter number of columns for matrix1 and number of rows for matrix2: ";
    cin>>n;

    cout<<"Enter number of columns for matrix2: ";
    cin>>k;

    //Declaration of host arrays
    float *hA = new float[m*n];
    float *hB = new float[n*k];
    float *hC = new float[m*k];

    //Declaration of device arrays
    float *dA, *dB, *dC;

    //Device Memory allocation 
    cudaMalloc((void **)&dA, m*n*sizeof(float));
    cudaMalloc((void **)&dB, n*k*sizeof(float));
    cudaMalloc((void **)&dC, m*k*sizeof(float));

    //Filling values for the matrices
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            hA[i*n+j] = i+j;
        }
    }

    for(int i=0; i<n; i++){
        for(int j=0; j<k; j++){
            hB[i*k+j] = i+j;
        }
    }

    //Copying values from host to device
    cudaMemcpy(dA, hA, m*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, n*k*sizeof(float), cudaMemcpyHostToDevice);

    //Calling the function for kernel launch
    launchKernel(dA, dB, dC, m, n, k);

    //Copying results from device to host
    cudaMemcpy(hC, dC, m*k*sizeof(float), cudaMemcpyDeviceToHost);

    //Print resultant array
    for(int i=0; i< m*k; i++){
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

