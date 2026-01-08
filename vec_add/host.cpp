#include "kernel.h"
#include <cstdlib>

int main(){
    int col ;

    // getting user input for number of columns
    cout<<"ENTER NUMBER OF COLUMNS : ";
    cin>>col;

    // allocating the required variables for the arrays
    float *arr1 = new float[col];
    float *arr2 = new float[col];
    float *cudaArr1;
    float *cudaArr2;
    float *res;
    float *host_res;

    // allocating memory for the device and host arrays
    cudaMalloc((void**) &cudaArr1, col*sizeof(float));
    cudaMalloc((void**) &cudaArr2, col*sizeof(float));
    cudaMalloc((void**) &res, col*sizeof(float));
    cudaMallocHost(&host_res, col*sizeof(float));
    
    // filling the matrix with values
    for (int j = 0; j<col; j++){
            arr1[j] = j;
            arr2[j] = j;
        }
    
    // copying data from host array to device array
    cudaMemcpy(cudaArr1, arr1, col*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaArr2, arr2, col*sizeof(float), cudaMemcpyHostToDevice);        
    
    // calling the function for launching the kernel
    mat_add_launch(cudaArr1, cudaArr2, res, col);
    
    // copying the result from device to host array
    cudaMemcpy(host_res, res, col*sizeof(float), cudaMemcpyDeviceToHost);

    // printing the results for verification
    for (int i = 0; i<col; i++)
        cout<<host_res[i]<<"  ";

    cout<<'\n';
    cudaDeviceSynchronize();

    // freeing the resources allocated
    cudaFree(cudaArr1);
    cudaFree(cudaArr2);
    cudaFree(res);
}