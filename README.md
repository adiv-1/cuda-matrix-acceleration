# CUDA Matrix Acceleration and Performance Engineering

## Overview

This project explores GPU acceleration for large-scale linear algebra workloads, with a focus on matrix multiplication (GEMM). The goal was to understand, implement, and benchmark different execution strategies across CPU and GPU architectures, and to evaluate how performance scales with problem size.

Rather than relying solely on high-level libraries, I implemented multiple versions of matrix multiplication from scratch, progressively optimizing the CUDA kernels and comparing them against NVIDIA's cuBLAS implementation. I also exposed a custom CUDA kernel as a shared library and integrated it into Python to demonstrate practical interoperability.

The project emphasizes performance analysis, architectural trade-offs, and low-level optimization techniques commonly used in high-performance computing and machine learning systems.

### Motivation

Modern machine learning and scientific computing workloads are dominated by linear algebra. Understanding how these operations map to hardware, and how performance changes across CPU and GPU implementations is essential for building efficient systems.

This project was an exercise in performance engineering: starting from a naïve implementation, identifying bottlenecks, applying architectural optimizations, and benchmarking against production-grade libraries like cuBLAS.

---

## Objectives

- Implement baseline matrix multiplication on CPU (C and Python)
- Port the implementation to CUDA (naïve kernel)
- Optimize CUDA using shared-memory tiling
- Benchmark CPU vs GPU performance across increasing matrix sizes
- Compare custom kernels against cuBLAS
- Analyze scaling behavior and overhead
- Build a CUDA shared library (`.so`) and call it from Python using `ctypes`
- Extend the library with GPU-accelerated 2D convolution for image processing

---

## Implementations

### 1. CPU Baseline (C and Python)

The CPU baseline includes both C and Python reference implementations for dense matrix-matrix (GEMM) and matrix-vector (MatVec) workloads.

**Matrix Multiply (C)**

- Path: `cpu/matrix-cpu/matrix_cpu.c`
- Description: Standard triple-loop GEMM (float), prints execution time
- Build & Run:
  ```bash
  gcc -O2 cpu/matrix-cpu/matrix_cpu.c -o cpu/matrix-cpu/matrix_cpu
  ./cpu/matrix-cpu/matrix_cpu 1024
  ```

**Matrix Multiply (Python)**

- Path: `cpu/matrix-cpu/matrix_cpu.py`
- Run:
  ```bash
  python3 cpu/matrix-cpu/matrix_cpu.py
  ```

**Matrix-Vector Multiply (C)**

- Path: `cpu/matvec-cpu/matvec_cpu.c`
- Description: Standard matvec (float), prints execution time
- Build & Run:
  ```bash
  gcc -O2 cpu/matvec-cpu/matvec_cpu.c -o cpu/matvec-cpu/matvec_cpu
  ./cpu/matvec-cpu/matvec_cpu 1024
  ```

**Matrix-Vector Multiply (Python)**

- Path: `cpu/matvec-cpu/matvec_cpu.py`
- Run:
  ```bash
  python3 cpu/matvec-cpu/matvec_cpu.py
  ```

**Observations:**

- Python matvec performance is reasonable for moderate sizes, but Python GEMM is substantially slower than C due to Python's interpreter overhead
- The C implementations served as the CPU performance baseline when measuring GPU speedups
- As matrix size grows, CPU execution time increases dramatically (e.g., ~10 minutes for N=4096)

---

### 2. Naïve CUDA Kernel

**What I Implemented:**

- Each GPU thread computes one output element `C[row,col]`
- Inner loop over k accumulates `A[row,k] * B[k,col]` into a scalar sum
- Uses only global memory (no shared memory optimization)

**Host Workflow:**

1. Allocate and initialize host arrays A, B, C with random floats
2. `cudaMalloc` device buffers, `cudaMemcpy` A and B to device
3. Configure launch: threads = `dim3(16,16)`, blocks = `(N+15)/16` in each dimension
4. Launch kernel and measure kernel time with `cudaEvent_t` (start/stop)
5. `cudaMemcpy` result back to host, free device/host memory

**Build & Run:**

```bash
nvcc -arch=sm_75 matrix_gpu_naive.cu -o matrix_gpu_naive
./matrix_gpu_naive 512
./matrix_gpu_naive 1024
./matrix_gpu_naive 2048
```

**Observations:**

- First run (N=512) takes longest despite being the smallest matrix due to GPU initialization overhead (memory allocation, kernel loading, GPU clock ramp-up)
- Later runs reuse the already-initialized CUDA context, so they're faster despite larger N
- Parallelism significantly reduces execution time compared to CPU, even in this unoptimized version
- Memory-bound and lacks shared-memory tiling; significant room for improvement

---

### 3. Tiled CUDA Kernel (Shared Memory Optimization)

**Optimization Strategy:**

- Use `__shared__` memory: Each block loads `TILE×TILE` sub-tiles of A and B into shared arrays (`tile_a`, `tile_b`)
- Each thread loads one element of A and one of B into shared memory, then calls `__syncthreads()` to ensure all threads have loaded their data
- Inner loop multiplies the loaded tiles (k = 0 to TILE-1), accumulating into a per-thread sum—this reuses values from fast shared memory instead of repeatedly reading from slow global memory
- Boundary checks handle non-multiples of TILE by writing zeros for out-of-range elements

**Launch Configuration:**

- Blocks of `TILE×TILE` threads (typically 16×16)
- Grid covers entire matrix via ceil-divide: `(N + TILE - 1) / TILE`

**Build & Run:**

```bash
nvcc -arch=sm_75 matrix_gpu_tiled.cu -o matrix_gpu_tiled
./matrix_gpu_tiled 512
./matrix_gpu_tiled 1024
./matrix_gpu_tiled 2048
```

**Key Benefits:**

- Far fewer global memory loads
- Higher arithmetic intensity and better memory coalescing
- Significant speedup for large N (typically 1.5-4× faster than naïve CUDA)
- Still lightweight compared to production libraries; next steps could include register blocking, loop unrolling, or GPU-specific TILE tuning

---

### 4. cuBLAS (Production Baseline)

**What is cuBLAS:**

- NVIDIA's highly optimized BLAS (Basic Linear Algebra Subprograms) library
- Called `cublasSgemm` for single-precision GEMM: `C = alpha*A*B + beta*C`

**Why it's Faster:**

- Architecture-specific optimizations for each GPU generation
- Advanced tiling strategies beyond simple shared memory blocking
- Instruction-level parallelism and register-level optimizations
- Low-level hardware features (tensor cores on newer GPUs, optimal warp scheduling)
- Extensively tested and continuously tuned by NVIDIA engineers

**Workflow:**

1. Create cuBLAS handle with `cublasCreate(&handle)`
2. Allocate/copy matrices to device
3. Call `cublasSgemm` with `alpha=1.0` and `beta=0.0`
4. Measure time using CUDA events
5. Copy result back to host

**Build & Run:**

```bash
nvcc matrix_gpu_cublas.cu -o matrix_gpu_cublas -lcublas
./matrix_gpu_cublas 512
./matrix_gpu_cublas 1024
./matrix_gpu_cublas 2048
```

**Observations:**

- For small matrices (N=512, 1024), hand-optimized kernel can be competitive due to lower overhead
- For large matrices (N≥2048), cuBLAS drastically outperforms custom kernels
- As N increases, cuBLAS scales much more efficiently, demonstrating the value of production-grade optimization

---

### 5. CUDA Shared Library + Python Integration

**Creating the Shared Library:**

The library includes a tiled matrix multiplication kernel wrapped in a C-style function for Python compatibility:

```bash
nvcc -Xcompiler -fPIC -shared matrix_lib.cu -o libmatrix.so
```

**Python Integration (ctypes):**

- Loaded library using `ctypes.CDLL('./libmatrix.so')`
- Defined function signatures with `argtypes` to specify NumPy array types
- Passed NumPy arrays (raveled to 1D, `dtype=float32`) directly to GPU function
- Measured end-to-end time including host-device transfers

**Example Usage:**

```python
import ctypes
import numpy as np

lib = ctypes.cdll.LoadLibrary("./libmatrix.so")
lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int
]

N = 1024
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), N)
```

**Observations:**

- Python overhead is minimal once arrays are allocated
- Transfer time dominates for small matrices
- Results comparable to running compiled CUDA code directly
- Enables integration into data science workflows without sacrificing GPU performance

---

### 6. 2D Convolution (GPU-Accelerated Image Processing)

**CUDA Kernel (`convolution2D_GPU`):**

- Each thread computes convolution for one output pixel
- Applies a 3×3 filter with boundary checking
- Example edge detection kernel:
  ```
  -1 -1 -1
  -1  8 -1
  -1 -1 -1
  ```

**Workflow:**

1. Allocate random grayscale image (512×512, `uint8`)
2. Define 3×3 filter kernel
3. Copy image and kernel to device
4. Launch 16×16 thread blocks
5. Measure kernel time with CUDA events
6. Copy result back to host

**Build & Run:**

```bash
nvcc conv_gpu.cu -o conv_gpu
./conv_gpu
```

**Python Integration:**

Added `gpu_convolution` to the shared library:

```bash
nvcc -Xcompiler -fPIC -shared matrix_lib.cu conv_lib.cu -o libmatrix.so
```

Tested with doubling image sizes (512×512 → 8192×8192) to evaluate scaling.

**Observations:**

- GPU convolution significantly faster than CPU for large images
- Python timing includes transfer overhead; direct CUDA timing shows pure kernel performance
- Demonstrates how custom CUDA functions can be added to shared libraries for specific domain tasks

---

## Performance Results

### Matrix Multiplication Performance

| Implementation     | N=512   | N=1024  | N=2048   | N=4096    | N=8192     |
| ------------------ | ------- | ------- | -------- | --------- | ---------- |
| **CPU (C)**        | 0.49 s  | 3.71 s  | 37.63 s  | ~10 min   | -          |
| **Naïve CUDA**     | 1.22 ms | 9.24 ms | 74.93 ms | 398.44 ms | 2651.64 ms |
| **Optimized CUDA** | 0.32 ms | 2.20 ms | 46.33 ms | 292.06 ms | 1739.15 ms |
| **cuBLAS**         | 5.49 ms | 6.14 ms | 11.47 ms | 53.74 ms  | 293.30 ms  |

### Speedup Analysis (CPU time / GPU time)

| Implementation     | N=512 | N=1024 | N=2048 | N=4096 |
| ------------------ | ----- | ------ | ------ | ------ |
| **Naïve CUDA**     | 401×  | 402×   | 502×   | 1506×  |
| **Optimized CUDA** | 1517× | 1688×  | 812×   | 2054×  |

---

## Key Findings & Analysis

### 1. Performance Scaling with Matrix Size

As matrix size N increases, runtime increases significantly across all implementations:

- **CPU**: Scales very poorly—grows from under 1 second at N=512 to approximately 10 minutes at N=4096
- **GPU**: Runtimes also increase with N, but remain several orders of magnitude faster (hundreds to thousands of times faster than CPU)
- **Conclusion**: GPU's superior ability to handle large-scale parallel computation becomes more pronounced as problem size grows

### 2. When Does GPU Outperform CPU?

- GPU already significantly outperforms CPU at N=512 (~400× faster)
- The gap widens as N increases (up to ~2000× faster at N=4096)
- **Key insight**: GPU advantages become more pronounced at larger N, but even moderate-sized problems benefit from GPU acceleration

### 3. Impact of Tiling Optimization

Tiling optimization provides noticeable speedup over naïve CUDA across all tested matrix sizes:

- **Smaller matrices**: Approximately 3-4× faster than naïve version
- **Larger matrices**: Speedups range from 1.3× to 2×
- **Why it helps**: More efficient memory access patterns and better use of shared memory reduce global memory latency

### 4. cuBLAS vs. Hand-Optimized Kernels

- **Small matrices (N=512, 1024)**: Optimized CUDA kernel outperforms cuBLAS due to lower overhead and simpler launch costs
- **Large matrices (N≥2048)**: cuBLAS drastically outperforms optimized kernel
- **As N increases**: cuBLAS scales much more efficiently, demonstrating superior optimization for large workloads
- **Why cuBLAS wins**: Architecture-specific optimizations, advanced tiling strategies, instruction-level parallelism, and low-level hardware features that are difficult to replicate in hand-written kernels

### 5. Python Integration Overhead

- Python overhead is minimal once NumPy arrays are allocated
- Transfer time between host and device introduces measurable overhead, particularly for small matrices
- For production workflows, batching operations or keeping data on GPU can amortize transfer costs

---

## Running CUDA on Google Cloud

For this project, I tested CUDA performance using Google Colab Pro (provides free access to Tesla T4 GPUs for students). Alternatively, you can provision a GPU-enabled VM on Google Cloud Compute Engine:

**Provisioning a GPU Instance:**

1. Create VM: GPU (Tesla T4 or V100), machine type n1-standard-8 or larger, Ubuntu 20.04
2. Install drivers and CUDA toolkit:
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-driver-470
   sudo apt-get install -y nvidia-cuda-toolkit
   nvcc --version
   ```
3. Compile and run:
   ```bash
   nvcc matrix_gpu_naive.cu -o matrix_gpu_naive
   ./matrix_gpu_naive 512
   ./matrix_gpu_naive 1024
   ./matrix_gpu_naive 2048
   ```

**Using Google Colab:**

- Colab Pro provides Tesla T4 GPUs with minimal setup
- Simply upload your `.cu` files and compile/run in notebook cells
- Ideal for quick experimentation without managing cloud infrastructure

---

## Key Takeaways

1. **GPU acceleration is most effective for sufficiently large workloads** where kernel execution amortizes transfer overhead
2. **Memory hierarchy matters**: Global vs shared memory is a dominant factor in CUDA performance
3. **Hand-optimized kernels can be competitive** but rarely match production libraries like cuBLAS
4. **Shared libraries enable practical integration**: Wrapping CUDA kernels as `.so` files allows integration into Python workflows without sacrificing performance
5. **Production libraries are highly optimized**: cuBLAS demonstrates the value of architecture-specific tuning and continuous optimization by hardware vendors
6. **Understanding bottlenecks is crucial**: Moving from naïve implementations to optimized kernels requires understanding memory access patterns, occupancy, and hardware capabilities

---

## Future Extensions

- **Multi-GPU scaling**: Distribute computation across multiple GPUs using NCCL or custom P2P transfers
- **Mixed precision**: Leverage FP16 and Tensor Cores on Ampere/Hopper GPUs for even higher throughput
- **Asynchronous memory transfers**: Overlap computation and data movement using CUDA streams
- **Advanced profiling**: Use Nsight Compute to identify bottlenecks and optimize kernel performance
- **Batched GEMM**: Process multiple smaller matrices simultaneously to improve GPU utilization
- **Integration with ML frameworks**: Expose kernels to PyTorch/TensorFlow for custom layer implementations

---

## Repository Structure

```
cuda-matrix-acceleration/
├── cpu/
│   ├── matrix-cpu/
│   │   ├── matrix_cpu.c
│   │   └── matrix_cpu.py
│   └── matvec-cpu/
│       ├── matvec_cpu.c
│       └── matvec_cpu.py
├── gpu_cuda/
│   ├── matrix_gpu_naive.cu
│   ├── matrix_gpu_tiled.cu
│   ├── matrix_gpu_cublas.cu
│   ├── matrix_lib.cu
│   ├── conv_gpu.cu
│   └── conv_lib.cu
└── README.md
```

---

## Acknowledgments

This project was developed as part of a performance engineering exploration into GPU acceleration for linear algebra workloads. Special thanks to the NVIDIA documentation and academic resources that provided insights into CUDA optimization techniques.

---

## License

This project is provided for educational purposes. Feel free to use and modify the code for learning and research.
