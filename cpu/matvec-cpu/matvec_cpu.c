# include <stdio.h>
# include <stdlib.h>
# include <time.h>

 // naive matrix-vector multiply: y = A * x
void matvecMultiplyCPU(float *A, float *x, float *y, int N) {
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

int main(int argc, char **argv) {
    // default problem size
    int N = (argc > 1) ? atoi(argv[1]) : 512;

    // sizes for allocations
    size_t matrix_size = N * N * sizeof(float);
    size_t vector_size = N * sizeof(float);

    // allocate buffers
    float *A = (float *)malloc(matrix_size);
    float *x = (float *)malloc(vector_size);
    float *y = (float *)malloc(vector_size);

    // initialize inputs
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 100 / 100.0f;
    }

    for (int i = 0; i < N; i++) {
        x[i] = rand() % 100 / 100.0f;
    }

    // time the CPU matvec
    clock_t start = clock();
    matvecMultiplyCPU(A, x, y, N);
    clock_t end = clock();

    volatile float sink = y[0]; // prevent optimization

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU execution time (N=%d): %f seconds\n", N, elapsed);

    // cleanup
    free(A);
    free(x);
    free(y);
    return 0;
}