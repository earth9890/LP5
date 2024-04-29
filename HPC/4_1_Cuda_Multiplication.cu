#include <bits/stdc++.h> // Includes all the standard C++ libraries
#include <cuda.h>        // CUDA header file

#define BLOCK_SIZE 16 // Define block size for CUDA kernels

/*
Name : Harish Sugandhi , Saurabh Shete
*/

using namespace std;

// CUDA kernel to perform matrix multiplication
__global__ void multiply(int *a, int *b, int *c, int c_rows, int common, int c_cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Calculate row index in the result matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Calculate column index in the result matrix
    int sum = 0;

    // Check if the thread is within the matrix dimensions
    if (col < c_cols && row < c_rows)
    {
        // Perform the dot product of the row of A and column of B to compute the element of C
        for (int j = 0; j < common; j++)
        {
            sum += a[row * common + j] * b[j * c_cols + col];
        }
        // Store the result in the output matrix
        c[c_cols * row + col] = sum;
    }
}

// Function to initialize a matrix with random values
void initialize(int *a, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            a[i * cols + j] = rand() % 10; // Generate random values between 0 and 9
        }
    }
}

// Function to print a matrix
void print(int *a, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            cout << a[i * cols + j] << " "; // Print each element of the matrix
        }
        cout << "\n"; // Move to the next row
    }
}

int main()
{
    // Define matrix dimensions
    int A_rows = 3, A_cols = 2, B_rows = 2, B_cols = 4, C_rows = A_rows, C_cols = B_cols;

    // Calculate matrix sizes
    int A_size = A_rows * A_cols;
    int B_size = B_rows * B_cols;
    int C_size = C_rows * C_cols;

    // Declare pointers for host and device memory
    int *A, *B, *C;
    int *m1, *m2, *m3;

    // Allocate memory for matrices on host (CPU)
    A = new int[A_size];
    B = new int[B_size];
    C = new int[C_size];

    // Allocate memory for matrices on device (GPU)
    cudaMalloc(&m1, A_size * sizeof(int));
    cudaMalloc(&m2, B_size * sizeof(int));
    cudaMalloc(&m3, C_size * sizeof(int));

    // Initialize matrices A and B with random values
    initialize(A, A_rows, A_cols);
    initialize(B, B_rows, B_cols);

    // Print matrices A and B
    cout << "Matrix A:- \n";
    print(A, A_rows, A_cols);

    cout << "Matrix B:- \n";
    print(B, B_rows, B_cols);

    // Copy matrices A and B from host to device
    cudaMemcpy(m1, A, A_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(m2, B, B_size * sizeof(int), cudaMemcpyHostToDevice);

    // Define CUDA grid and block dimensions
    dim3 dimGrid((A_rows + BLOCK_SIZE - 1) / BLOCK_SIZE, (B_cols + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // Launch CUDA kernel to perform matrix multiplication
    multiply<<<dimGrid, dimBlock>>>(m1, m2, m3, C_rows, A_cols, C_cols);

    // Copy result matrix C from device to host
    cudaMemcpy(C, m3, C_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result matrix C
    cout << "Matrix C:- \n";
    print(C, C_rows, C_cols);

    // Free host memory
    delete[] A;
    delete[] B;
    delete[] C;

    // Free device memory
    cudaFree(m1);
    cudaFree(m2);
    cudaFree(m3);
}

// nvcc 4_1_Cuda_Multiplication.cu -o 4_1_Cuda_Multiplication

/*
    This program demonstrates matrix multiplication using CUDA.
    CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA.
    It allows developers to use CUDA-enabled GPUs for general-purpose processing, which can significantly accelerate compute-intensive tasks.
    In this program:
    - CUDA kernels are used to perform matrix multiplication on the GPU.
    - The matrices A and B are initialized with random values on the CPU.
    - The matrices A and B are then copied from the host (CPU) to the device (GPU).
    - CUDA kernels are launched to perform the matrix multiplication.
    - The result matrix C is copied from the device to the host.
    - Finally, the result matrix C is printed.
*/
