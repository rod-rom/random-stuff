#include <stdio.h>

__global__ void matrixMultiply(const float* A, const float* B, float* C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;    

    if (row < N && col < N){
        float sum = 0.0f;
        for (int i = 0; i < N; ++i){
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(void){
    const int N = 3; // Size of matrices (N x N)

    // Allocate memory for input and output matrices on the Host (CPU)
    // float* h_A = (float*)malloc(N * N * sizeof(float));
    // float* h_B = (float*)malloc(N * N * sizeof(float));
    float* h_C = (float*)malloc(N * N * sizeof(float));

    // Initialize input matrices h_A and h_B
    float h_A[] = {1, 0, 0,
                    0, 1, 0,
                    0, 0, 1};

    float h_B[] = { 2, 5, 1,
                    6, 7, 1,
                    1, 8, 1};

    // Allocate memory for input and output matrices on the Device (GPU)
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for launching the kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x -1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch the matric multiplcation kernel
    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Copy the result matrix from device to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result matrix
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
            printf("%f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}