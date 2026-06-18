#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Part 4: CUDA SIMT implementation for Bilateral Filter
//
// This CUDA program only runs the GPU implementation.  The standalone CPU
// baseline is provided separately in main_cpu.cpp, following the TA example
// style.
//
// Default execution:
//   ./main
// Optional execution:
//   ./main <input_txt> <output_txt> [iterations] [threads_per_block]
//
// CUDA mapping:
//   one CUDA thread computes one output pixel.
// -----------------------------------------------------------------------------

#define DEFAULT_INPUT_FILE  "../test/cyberpunk2077_in.txt"
#define DEFAULT_OUTPUT_FILE "output/cuda_out.txt"

#define RADIUS 4
#define WINDOW_SIZE (2 * RADIUS + 1)
#define KERNEL_ELEMS (WINDOW_SIZE * WINDOW_SIZE)
#define DEFAULT_ITERATIONS 3
#define DEFAULT_THREADS_PER_BLOCK 256

#define SPATIAL_ALPHA 0.0600f
#define RANGE_ALPHA   0.0009f

__constant__ float c_spatial[KERNEL_ELEMS];
__constant__ int   c_dxs[KERNEL_ELEMS];
__constant__ int   c_dys[KERNEL_ELEMS];

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA ERROR at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1); \
    } \
} while (0)

__device__ __forceinline__ int clamp_int_device(int v, int lo, int hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

static int load_image_txt(const char *path, float **image, int *width, int *height) {
    FILE *fp = std::fopen(path, "r");
    if (!fp) {
        std::printf("ERROR: cannot open input file %s\n", path);
        return 0;
    }

    if (std::fscanf(fp, "%d %d", width, height) != 2) {
        std::printf("ERROR: invalid input header in %s\n", path);
        std::fclose(fp);
        return 0;
    }

    const int n = (*width) * (*height);
    *image = (float *)std::malloc(sizeof(float) * n);
    if (!(*image)) {
        std::printf("ERROR: malloc failed for input image\n");
        std::fclose(fp);
        return 0;
    }

    for (int i = 0; i < n; i++) {
        int pixel = 0;
        if (std::fscanf(fp, "%d", &pixel) != 1) {
            std::printf("ERROR: not enough pixel values in %s\n", path);
            std::free(*image);
            *image = NULL;
            std::fclose(fp);
            return 0;
        }
        if (pixel < 0) pixel = 0;
        if (pixel > 255) pixel = 255;
        (*image)[i] = (float)pixel;
    }

    std::fclose(fp);
    return 1;
}

static void write_output_txt(const char *path, const float *image, int width, int height) {
    FILE *fp = std::fopen(path, "w");
    if (!fp) {
        std::printf("WARNING: cannot write output file %s\n", path);
        return;
    }

    std::fprintf(fp, "%d %d\n", width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pixel = (int)(image[y * width + x] + 0.5f);
            if (pixel < 0) pixel = 0;
            if (pixel > 255) pixel = 255;
            std::fprintf(fp, "%d%c", pixel, (x == width - 1) ? '\n' : ' ');
        }
    }

    std::fclose(fp);
}

static void precompute_spatial_kernel(float spatial[KERNEL_ELEMS], int dxs[KERNEL_ELEMS], int dys[KERNEL_ELEMS]) {
    int k = 0;
    for (int dy = -RADIUS; dy <= RADIUS; dy++) {
        for (int dx = -RADIUS; dx <= RADIUS; dx++) {
            const float dist2 = (float)(dx * dx + dy * dy);
            dxs[k] = dx;
            dys[k] = dy;
            spatial[k] = 1.0f / (1.0f + SPATIAL_ALPHA * dist2);
            k++;
        }
    }
}

__global__ void bilateral_filter_cuda_kernel(
    const float *src,
    float *dst,
    int width,
    int height,
    int total_pixels
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    const int x = idx % width;
    const int y = idx / width;
    const float center = src[idx];

    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;

    #pragma unroll
    for (int k = 0; k < KERNEL_ELEMS; k++) {
        const int nx = clamp_int_device(x + c_dxs[k], 0, width - 1);
        const int ny = clamp_int_device(y + c_dys[k], 0, height - 1);
        const float neighbor = src[ny * width + nx];

        const float diff = neighbor - center;
        const float range_weight = 1.0f / (1.0f + RANGE_ALPHA * diff * diff);
        const float weight = c_spatial[k] * range_weight;

        weighted_sum += weight * neighbor;
        weight_sum += weight;
    }

    dst[idx] = weighted_sum / weight_sum;
}

static unsigned long long checksum_u8_from_float(const float *image, int n) {
    unsigned long long hash = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) {
        int pixel = (int)(image[i] + 0.5f);
        if (pixel < 0) pixel = 0;
        if (pixel > 255) pixel = 255;
        hash ^= (unsigned long long)pixel;
        hash *= 1099511628211ULL;
    }
    return hash;
}

static void compute_image_stats(
    const float *image,
    int n,
    double *sum_out,
    double *mean_out,
    double *mean_square_out,
    unsigned long long *checksum_out
) {
    double sum = 0.0;
    double energy = 0.0;

    for (int i = 0; i < n; i++) {
        sum += (double)image[i];
        energy += (double)image[i] * (double)image[i];
    }

    *sum_out = sum;
    *mean_out = sum / (double)n;
    *mean_square_out = energy / (double)n;
    *checksum_out = checksum_u8_from_float(image, n);
}

static void print_final_summary(
    const char *input_file,
    const char *output_file,
    int width,
    int height,
    int iterations,
    int threads_per_block,
    int cuda_blocks,
    float gpu_ms,
    double gpu_sum,
    double gpu_mean,
    double gpu_mean_square,
    unsigned long long gpu_checksum
) {
    const int n = width * height;
    const long long total_inner_loop_evaluations = (long long)n * KERNEL_ELEMS * iterations;
    const long long launched_threads_per_iteration = (long long)cuda_blocks * threads_per_block;

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("                  Execution Configuration\n");
    std::printf("============================================================\n");
    std::printf("implementation               = CUDA SIMT one-thread-per-pixel\n");
    std::printf("input_file                   = %s\n", input_file);
    std::printf("output_file                  = %s\n", output_file);
    std::printf("image_width                  = %d\n", width);
    std::printf("image_height                 = %d\n", height);
    std::printf("total_pixels                 = %d\n", n);
    std::printf("iterations                   = %d\n", iterations);

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("                   Algorithm Parameters\n");
    std::printf("============================================================\n");
    std::printf("algorithm                    = Bilateral Filter\n");
    std::printf("radius                       = %d\n", RADIUS);
    std::printf("window_size                  = %d x %d\n", WINDOW_SIZE, WINDOW_SIZE);
    std::printf("window_elems_per_pixel       = %d\n", KERNEL_ELEMS);
    std::printf("spatial_alpha                = %.4f\n", SPATIAL_ALPHA);
    std::printf("range_alpha                  = %.4f\n", RANGE_ALPHA);
    std::printf("boundary_policy              = clamp-to-edge\n");
    std::printf("range_kernel                 = 1 / (1 + RANGE_ALPHA * diff * diff)\n");

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("                       CUDA Mapping\n");
    std::printf("============================================================\n");
    std::printf("part                         = Part 4\n");
    std::printf("execution_model              = CUDA SIMT\n");
    std::printf("thread_mapping               = one CUDA thread = one output pixel\n");
    std::printf("threads_per_block            = %d\n", threads_per_block);
    std::printf("cuda_blocks                  = %d\n", cuda_blocks);
    std::printf("active_threads_per_iter      = %d\n", n);
    std::printf("launched_threads_per_iter    = %lld\n", launched_threads_per_iteration);
    std::printf("timing_method_gpu            = cudaEvent kernel-only timing\n");
    std::printf("memcpy_in_gpu_timing         = no\n");
    std::printf("file_io_in_timing            = no\n");
    std::printf("cpu_reference_program        = main_cpu.cpp\n");

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("                      Workload Summary\n");
    std::printf("============================================================\n");
    std::printf("total_inner_loop_evaluations = %lld\n", total_inner_loop_evaluations);
    std::printf("operations_per_neighbor      = range weight + weighted sum + weight sum\n");

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("                       Runtime Summary\n");
    std::printf("============================================================\n");
    std::printf("gpu_kernel_time_ms           = %.6f\n", gpu_ms);

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("                    Verification Output\n");
    std::printf("============================================================\n");
    std::printf("gpu_output_sum               = %.6f\n", gpu_sum);
    std::printf("gpu_output_mean              = %.6f\n", gpu_mean);
    std::printf("gpu_output_mean_square       = %.6f\n", gpu_mean_square);
    std::printf("gpu_output_checksum          = 0x%016llx\n", gpu_checksum);
    std::printf("compare_with_cpu_output      = run ./main_cpu and compare checksum/output_sum\n");

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("                       Output Status\n");
    std::printf("============================================================\n");
    std::printf("output_written               = %s\n", output_file);

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("                      Program Finished\n");
    std::printf("============================================================\n");
    std::printf("\n");
}

int main(int argc, char **argv) {
    const char *input_file = DEFAULT_INPUT_FILE;
    const char *output_file = DEFAULT_OUTPUT_FILE;
    int iterations = DEFAULT_ITERATIONS;
    int threads_per_block = DEFAULT_THREADS_PER_BLOCK;

    if (argc >= 2) input_file = argv[1];
    if (argc >= 3) output_file = argv[2];
    if (argc >= 4) iterations = std::atoi(argv[3]);
    if (argc >= 5) threads_per_block = std::atoi(argv[4]);

    if (iterations <= 0) {
        std::printf("ERROR: iterations must be positive.\n");
        return 1;
    }
    if (threads_per_block <= 0 || threads_per_block > 1024) {
        std::printf("ERROR: threads_per_block must be in 1..1024.\n");
        return 1;
    }

    float *input = NULL;
    int width = 0;
    int height = 0;
    if (!load_image_txt(input_file, &input, &width, &height)) {
        return 1;
    }

    const int n = width * height;
    const size_t bytes = (size_t)n * sizeof(float);

    float spatial[KERNEL_ELEMS];
    int dxs[KERNEL_ELEMS];
    int dys[KERNEL_ELEMS];
    precompute_spatial_kernel(spatial, dxs, dys);

    CUDA_CHECK(cudaMemcpyToSymbol(c_spatial, spatial, sizeof(float) * KERNEL_ELEMS));
    CUDA_CHECK(cudaMemcpyToSymbol(c_dxs, dxs, sizeof(int) * KERNEL_ELEMS));
    CUDA_CHECK(cudaMemcpyToSymbol(c_dys, dys, sizeof(int) * KERNEL_ELEMS));

    float *d_a = NULL;
    float *d_b = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_b, 0, bytes));

    const int cuda_blocks = (n + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    float *d_src = d_a;
    float *d_dst = d_b;
    for (int iter = 0; iter < iterations; iter++) {
        bilateral_filter_cuda_kernel<<<cuda_blocks, threads_per_block>>>(d_src, d_dst, width, height, n);
        CUDA_CHECK(cudaGetLastError());
        float *tmp = d_src;
        d_src = d_dst;
        d_dst = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    float *gpu_output = (float *)std::malloc(bytes);
    if (!gpu_output) {
        std::printf("ERROR: GPU output malloc failed\n");
        std::free(input);
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(gpu_output, d_src, bytes, cudaMemcpyDeviceToHost));

    double gpu_sum, gpu_mean, gpu_mean_square;
    unsigned long long gpu_checksum;
    compute_image_stats(gpu_output, n, &gpu_sum, &gpu_mean, &gpu_mean_square, &gpu_checksum);

    write_output_txt(output_file, gpu_output, width, height);

    print_final_summary(
        input_file,
        output_file,
        width,
        height,
        iterations,
        threads_per_block,
        cuda_blocks,
        gpu_ms,
        gpu_sum,
        gpu_mean,
        gpu_mean_square,
        gpu_checksum
    );

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    std::free(input);
    std::free(gpu_output);

    return 0;
}
