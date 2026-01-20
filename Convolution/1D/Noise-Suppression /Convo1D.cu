#include <cuda_runtime.h>
#include <iostream>

__global__ void convolution1D_naive(float *input, float *mask, float *output, int n, int maskWidth) {
    // Calculate global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (i >= n) return;

    // Output
    float pValue = 0.0f;
    
    // Radius
    int r = maskWidth / 2;
     
    // 'j' is the position inside the mask (0 to maskWidth-1)
    for (int j = 0; j < maskWidth; j++) {
        
        // Calculate the neighbor's index in the Input array
        // If i is our current pixel, and we want the neighbor defined by mask index j:
        // neighbor_index = i + (j - r)
        // Example: if j=0 (start of mask), offset is -r (left side)
        int neighborIdx = i + (j - r);

        // Handle Boundaries (Ghost Cells/Padding)
        // If the neighbor is outside the array (left of 0 or right of N), assume value is 0.
        float inputVal = 0.0f;
        if (neighborIdx >= 0 && neighborIdx < n) {
            inputVal = input[neighborIdx];
        }

        pValue += inputVal * mask[j];
    }

    output[i] = pValue;
}

int main() {
    int n = 8; 
    int bytes_n = n * sizeof(float);
    
    int maskWidth = 3;
    int bytes_m = maskWidth * sizeof(float);

    // The "Noisy Signal"
    float h_input[] = {1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0}; 

    // Kernel
    float h_mask[] = {0.25, 0.5, 0.25}; 

    float h_output[8];

    float *d_input, *d_mask, *d_output;
    cudaMalloc(&d_input, bytes_n);
    cudaMalloc(&d_mask, bytes_m);
    cudaMalloc(&d_output, bytes_n);

    cudaMemcpy(d_input, h_input, bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, bytes_m, cudaMemcpyHostToDevice);


    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    convolution1D_naive<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_mask, d_output, n, maskWidth);

    cudaMemcpy(h_output, d_output, bytes_n, cudaMemcpyDeviceToHost);
    
    printf("Input:  ");
    for(int i=0; i<n; i++) printf("%0.1f ", h_input[i]);
    printf("\nOutput: ");
    for(int i=0; i<n; i++) printf("%0.1f ", h_output[i]);
    printf("\n");

    // Free memory
    cudaFree(d_input); cudaFree(d_mask); cudaFree(d_output);
    return 0;
}
