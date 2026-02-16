import time
import random

# naive triple-loop GEMM (Python logic implementation)
def matrix_multiply(A, B, C, N):
    for i in range(N):
        for j in range(N):
            sum = 0
            for k in range(N):
                sum += A[i*N + k] * B[k*N + j]
            C[i*N + j] = sum

# initializing matrices with small random integers
def initialize_matrices(N):
    A = [random.randint(0, 10) for _ in range(N * N)]
    B = [random.randint(0, 10) for _ in range(N * N)]
    C = [0 for _ in range(N * N)]
    return A, B, C

# run simple timing for several sizes
def main():
    for N in [512, 1024, 2048]:
        A, B, C = initialize_matrices(N)
        start_time = time.time()
        matrix_multiply(A, B, C, N)
        end_time = time.time()
        print(f"Matrix multiplication of size {N}x{N} completed in {end_time - start_time:.6f} seconds.")

if __name__ == "__main__":
    main()