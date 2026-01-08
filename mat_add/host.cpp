#include "kernel.h"
#include <cstdlib>

int main(){
    int row ;
    int col ;

    // getting user input for number of rows and columns
    cout<<"ENTER NUMBER OF ROWS : ";
    cin>>row;

    cout<<"ENTER NUMBER OF COLUMNS : ";
    cin>>col;

    // allocating the required variables for the arrays
    float *arr1 = new float[row*col];
    float *arr2 = new float[row*col];
    float *cudaArr1;
    float *cudaArr2;
    float *res;
    float *host_res;

    // allocating memory for the device and host arrays
    cudaMalloc((void**) &cudaArr1, row*col*sizeof(float));
    cudaMalloc((void**) &cudaArr2, row*col*sizeof(float));
    cudaMalloc((void**) &res, row*col*sizeof(float));
    cudaMallocHost(&host_res, row*col*sizeof(float));
    
    // filling the matrix with values
    for (int i = 0; i<row; i++){
        for (int j = 0; j<col; j++){
            arr1[i*col+j] = i+j;
            arr2[i*col+j] = i+j;
        }
    }
    
    // copying data from host array to device array
    cudaMemcpy(cudaArr1, arr1, row*col*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaArr2, arr2, row*col*sizeof(float), cudaMemcpyHostToDevice);        
    
    // calling the function for launching the kernel
    mat_add_launch(cudaArr1, cudaArr2, res, row, col);
    
    // copying the result from device to host array
    cudaMemcpy(host_res, res, row*col*sizeof(float), cudaMemcpyDeviceToHost);

    // printing the results for verification
    for (int i = 0; i<row*col; i++)
        cout<<host_res[i]<<"  ";

    cout<<'\n';
    cudaDeviceSynchronize();

    // freeing the resources allocated
    cudaFree(cudaArr1);
    cudaFree(cudaArr2);
    cudaFree(res);
}