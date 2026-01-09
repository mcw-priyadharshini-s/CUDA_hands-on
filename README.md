## CUDA hands-on

The following GPU configuration was used for these experiments,<br>

**GPU**:  NVIDIA GeForce RTX 2080 Ti<br>
**Max thread per block**:  1024

### Steps

Execute the following steps on the GPU.

- Clone the repository
    ```
    gitclone https://github.com/mcw-priyadharshini-s/CUDA_hands-on.git
    ```
- Checkout to branch - priya
    ```
    git checkout priya
    ```

- Move to respective kernel folder
    ```
    cd <kernel_folder_name>
    ```
- Compile the CUDA kernel using the following command
    ```
    nvcc <kernel_name>.cu -o <kernel_name>.exe
    ```
- Run the generated executable
    ```
    ./<kernel_name>.exe
    ```

### Performance

- **Vector Addition**
    - Total element : 500 M
    - Time taken:
        - CPU: 1043.678000 ms
        - GPU: 10.731680 ms
    - Performance improvement: ~97× speedup


- **Matrix Multiplication**
    - Inputs
        - A matrix dimension: 1024x1024
        - B matrix dimension: 1024x1024
    - Output
        - C matrix dimension: 1024x1024
    - Time taken
        - CPU: 4664.02 ms
        - GPU: 1.69779 ms
    - Performance improvement: ~2747× speedup
    - Note
        - Gpu cuda kernel naive implementation is done using memory coalescing



- **Matrix Multiplication with tiling**
    - Tiled width : 64
    - Inputs
        - A matrix dimension: 1024x1024
        - B matrix dimension: 1024x1024
    - Output
        - C matrix dimension: 1024x1024
    - Time taken
        - GPU naive implementation: 1.68333 ms
        - GPU tiling implementation: 0.002048 ms
    - Performance improvement: ~822× speedup
    - Note
        - Both naive and gpu tiling implementation is done using memory coalescing
        - Gpu tiling implementation is done using shared memory