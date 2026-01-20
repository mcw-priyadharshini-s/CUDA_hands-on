#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>

// Kernel Settings
#define MASK_DIM 3
#define RADIUS 1

// Allocate Constant Memory for the Mask
__constant__ float d_mask[MASK_DIM * MASK_DIM];

__global__ void convolution2D(unsigned char *input, unsigned char *output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width) return;

    float pValue = 0.0f;

    // Sliding Window Loop
    for (int i = -RADIUS; i <= RADIUS; i++) {
        for (int j = -RADIUS; j <= RADIUS; j++) {
            
            // Neighbor coordinates
            int r_idx = row + i;
            int c_idx = col + j;

            // Boundary Check
            if (r_idx >= 0 && r_idx < height && c_idx >= 0 && c_idx < width) {
                // Fetch Input Pixel
                float pixel = (float)input[r_idx * width + c_idx];
                
                // Fetch Mask Weight (mapping -1..1 to 0..2)
                int maskIdx = (i + RADIUS) * MASK_DIM + (j + RADIUS);
                
                pValue += pixel * d_mask[maskIdx];
            }
        }
    }

    // Clamp values to valid pixel range (0-255)
    if (pValue < 0.0f) pValue = 0.0f;
    if (pValue > 255.0f) pValue = 255.0f;

    output[row * width + col] = (unsigned char)pValue;
}

int main() {
    // 1. READ DATA FROM FILE
    std::ifstream infile("img_data.txt");
    if (!infile.is_open()) {
        std::cerr << "Error: Run preprocess.py first!" << std::endl;
        return -1;
    }

    int width, height;
    infile >> width >> height; // Read header
    std::cout << "CUDA: Reading " << width << "x" << height << " image..." << std::endl;

    int numPixels = width * height;
    size_t bytes = numPixels * sizeof(unsigned char);

    // Use vector for automatic host memory management
    std::vector<int> temp_input(numPixels); // Temp storage for reading ints
    std::vector<unsigned char> h_input(numPixels);
    std::vector<unsigned char> h_output(numPixels);

    // Read pixels
    for(int i=0; i<numPixels; i++) {
        infile >> temp_input[i];
        h_input[i] = (unsigned char)temp_input[i];
    }
    infile.close();

    // 2. DEFINE MASK (Sobel Vertical Edge Detection)
    float h_mask[] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };

    // 3. GPU ALLOCATION
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // 4. COPY TO DEVICE
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mask, h_mask, sizeof(float) * MASK_DIM * MASK_DIM);

    // 5. LAUNCH KERNEL
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    convolution2D<<<blocks, threads>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize(); // Wait for GPU to finish

    // 6. COPY BACK
    cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    // 7. WRITE OUTPUT TO FILE
    std::ofstream outfile("out_data.txt");
    outfile << width << " " << height << "\n";
    for(int i=0; i<numPixels; i++) {
        outfile << (int)h_output[i] << " ";
    }
    outfile.close();

    std::cout << "CUDA: Processing Complete. Saved to 'out_data.txt'" << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
