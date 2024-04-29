%% cu
#include <bits/stdc++.h> // Include all standard C++ libraries
#include <cuda.h>        // CUDA header file

    using namespace std;

// CUDA kernel to add two vectors
__global__ void add(int *a, int *b, int *c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Calculate thread ID

    // Check if thread ID is within vector size
    if (tid < n)
    {
        c[tid] = a[tid] + b[tid]; // Add corresponding elements of vectors
    }
}

// Function to initialize a vector with random values
void initialize(int *a, int n)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = rand() % 100; // Generate random values between 0 and 99
    }
}

// Function to print a vector
void print(int *a, int n)
{
    for (int i = 0; i < n; i++)
    {
        cout << a[i] << " "; // Print each element of the vector
    }
}

int main()
{
    int n = 5; // Size of the vectors

    int size = n;
    size_t mem = size * sizeof(int); // Calculate memory size for vectors

    int *A, *B, *C; // Host vectors
    int *X, *Y, *Z; // Device vectors

    A = new int[size]; // Allocate memory for vector A on the host
    B = new int[size]; // Allocate memory for vector B on the host
    C = new int[size]; // Allocate memory for vector C on the host

    // Initialize vectors A and B with random values
    initialize(A, n);
    initialize(B, n);

    // Print vectors A and B
    cout << "Vector A:- \n";
    print(A, n);

    cout << "Vector B:- \n";
    print(B, n);

    // Allocate memory for vectors on the device
    cudaMalloc(&X, mem);
    cudaMalloc(&Y, mem);
    cudaMalloc(&Z, mem);

    // Copy vectors A and B from host to device
    cudaMemcpy(X, A, mem, cudaMemcpyHostToDevice);
    cudaMemcpy(Y, B, mem, cudaMemcpyHostToDevice);

    int threads = 256;                                 // Number of threads per block
    int blocksPerThread = (n + threads - 1) / threads; // Calculate number of blocks needed

    // Launch CUDA kernel to add vectors
    add<<<blocksPerThread, threads>>>(X, Y, Z, n);

    // Copy result vector C from device to host
    cudaMemcpy(C, Z, mem, cudaMemcpyDeviceToHost);

    // Print the result vector C
    cout << "Vector C:- ";
    print(C, n);

    // Free host memory
    delete[] A;
    delete[] B;
    delete[] C;

    // Free device memory
    cudaFree(X);
    cudaFree(Y);
    cudaFree(Z);
}


// nvcc 4_Cuda_Add.cu -o 4_Cuda_Add