#include "kernel.h"

int main(){

    //Getting dimensions for matrices from the user
    int row, col;

    cout<<"Enter number of rows: ";
    cin>>row;

    cout<<"Enter number of columns: ";
    cin>>col;

    //Declaration of host arrays
    float *hA = new float[row*col];
    float *hB = new float[row*col];
    float *hC = new float[row*col];

    //Declaration of device arrays
    float *dA, *dB, *dC;

    size_t size = row*col*sizeof(float);


    //Device Memory allocation 
    cudaMalloc((void **)&dA, size);
    cudaMalloc((void **)&dB, size);
    cudaMalloc((void **)&dC, size);

    //Filling values for the matrices
    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            hA[i*col+j] = i+j;
            hB[i*col+j] = i+j;
        }
    }

    //Copying values from host to device
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    //Calling the function for kernel launch
    launchKernel(dA, dB, dC, row, col);

    //Copying results from device to host
    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

    //Print resultant array
    for(int i=0; i< row*col; i++){
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

