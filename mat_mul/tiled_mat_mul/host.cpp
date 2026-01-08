#include "kernel.h"
#include <cstdlib>

int main(){
    int m ;
    int n ;
    int k;

    // getting user input for number of rows and columns
    cout<<"ENTER NUMBER OF ROWS for arr1: ";
    cin>>m;

    cout<<"ENTER NUMBER OF COLUMNS for arr1: ";
    cin>>n;

    cout<<"ENTER NUMBER OF COLUMNS for arr2: ";
    cin>>k;

    // allocating the required variables for the arrays
    float *arr1 = new float[m*n];
    float *arr2 = new float[n*k];
    float *cudaArr1;
    float *cudaArr2;
    float *res;
    float *host_res;

    // allocating memory for the device and host arrays
    cudaMalloc((void**) &cudaArr1, m*n*sizeof(float));
    cudaMalloc((void**) &cudaArr2, n*k*sizeof(float));
    cudaMalloc((void**) &res, m*k*sizeof(float));
    cudaMallocHost(&host_res, m*k*sizeof(float));
    
    // filling the matrix with values
    for (int i = 0; i<m; i++){
        for (int j = 0; j<n; j++){
            arr1[i*n+j] = i+j;
        }
    }

    for (int i = 0; i<n; i++){
        for (int j = 0; j<k; j++){
            arr2[i*k+j] = i+j;
        }
    }
    
    
    // copying data from host array to device array
    cudaMemcpy(cudaArr1, arr1, m*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaArr2, arr2, n*k*sizeof(float), cudaMemcpyHostToDevice);        
    
    // calling the function for launching the kernel
    mat_mul_launch(cudaArr1, cudaArr2, res, m, n, k);
    
    // copying the result from device to host array
    cudaMemcpy(host_res, res, m*k*sizeof(float), cudaMemcpyDeviceToHost);

    // printing the results for verification
    for (int i = 0; i<m*k; i++)
        cout<<host_res[i]<<"  ";

    cout<<'\n';
    cudaDeviceSynchronize();

    // freeing the resources allocated
    cudaFree(cudaArr1);
    cudaFree(cudaArr2);
    cudaFree(res);
}