#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Part 5: CUDA multi-pattern 2D-grid implementation for Bilateral Filter
//
// This CUDA program only runs the GPU implementation.  The standalone CPU
// multi-pattern baseline is provided separately in main_cpu.cpp.
//
// Default execution:
//   ./main
// Optional execution:
//   ./main <input_txt> <output_txt_pattern0> [iterations] [patterns] [threads_per_block] [shared|global]
//
// CUDA mapping:
//   blockIdx.x / threadIdx.x -> output pixel index
//   blockIdx.y               -> pattern index
//   one CUDA thread computes one output pixel of one pattern.
//
// Pattern policy:
//   pattern 0 is exactly the original P4 input image.
//   pattern 1..N-1 are deterministic variants of the same base image.
// -----------------------------------------------------------------------------

#define MAX_RADIUS 4
#define MAX_WINDOW ((2 * MAX_RADIUS + 1) * (2 * MAX_RADIUS + 1))
#define DEFAULT_INPUT_FILE  "../test/cyberpunk2077_in.txt"
#define DEFAULT_OUTPUT_FILE "output/multi_cuda_out.txt"
#define DEFAULT_ITERATIONS 3
#define DEFAULT_PATTERNS 32
#define DEFAULT_THREADS_PER_BLOCK 256

#define SPATIAL_ALPHA 0.0600f
#define RANGE_ALPHA   0.0009f

__constant__ float c_spatial[MAX_WINDOW];
__constant__ int   c_dxs[MAX_WINDOW];
__constant__ int   c_dys[MAX_WINDOW];

static inline float clampf_host(float x, float lo, float hi) {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

static inline void cudaCheck(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        std::exit(1);
    }
}

static bool read_txt_image(const char *path, int &width, int &height, std::vector<float> &img) {
    std::ifstream fin(path);
    if (!fin) {
        std::fprintf(stderr, "Failed to open input file: %s\n", path);
        return false;
    }

    fin >> width >> height;
    if (!fin || width <= 0 || height <= 0) {
        std::fprintf(stderr, "Invalid image header in: %s\n", path);
        return false;
    }

    img.resize((size_t)width * height);
    for (int i = 0; i < width * height; i++) {
        fin >> img[i];
        if (!fin) {
            std::fprintf(stderr, "Invalid pixel data in: %s\n", path);
            return false;
        }
    }

    return true;
}

static bool write_txt_image(const char *path, int width, int height, const std::vector<float> &img) {
    std::ofstream fout(path);
    if (!fout) {
        std::fprintf(stderr, "Failed to open output file: %s\n", path);
        return false;
    }

    fout << width << " " << height << "\n";
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pixel = (int)std::lround(img[(size_t)y * width + x]);
            if (pixel < 0) pixel = 0;
            if (pixel > 255) pixel = 255;
            fout << pixel;
            if (x + 1 != width) fout << " ";
        }
        fout << "\n";
    }

    return true;
}

static uint64_t checksum_u8_rounded_range(const std::vector<float> &img, size_t begin, size_t count) {
    uint64_t h = 1469598103934665603ULL;

    for (size_t i = 0; i < count; i++) {
        int v = (int)std::lround(img[begin + i]);
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        h ^= (uint64_t)(uint8_t)v;
        h *= 1099511628211ULL;
    }

    return h;
}

static uint64_t checksum_u8_rounded_all(const std::vector<float> &img) {
    return checksum_u8_rounded_range(img, 0, img.size());
}

static void prepare_spatial_kernel(
    int radius,
    std::vector<float> &spatial,
    std::vector<int> &dxs,
    std::vector<int> &dys
) {
    spatial.clear();
    dxs.clear();
    dys.clear();

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            float dist2 = (float)(dx * dx + dy * dy);
            float w = 1.0f / (1.0f + SPATIAL_ALPHA * dist2);
            spatial.push_back(w);
            dxs.push_back(dx);
            dys.push_back(dy);
        }
    }
}

static void generate_patterns_from_base(
    const std::vector<float> &base,
    int width,
    int height,
    int patterns,
    std::vector<float> &all_patterns
) {
    const int image_size = width * height;
    all_patterns.resize((size_t)patterns * image_size);

    for (int p = 0; p < patterns; p++) {
        for (int i = 0; i < image_size; i++) {
            float v = base[i];

            if (p == 0) {
                all_patterns[(size_t)p * image_size + i] = v;
            } else {
                int x = i % width;
                int y = i / width;

                float shift = (float)(((p * 7) % 21) - 10);
                float scale = 1.0f + 0.02f * (float)((p % 5) - 2);
                float noise = (float)(((x * 13 + y * 17 + p * 19) % 7) - 3);
                float out = (v - 128.0f) * scale + 128.0f + shift + noise;
                all_patterns[(size_t)p * image_size + i] = clampf_host(out, 0.0f, 255.0f);
            }
        }
    }
}

static int compute_shared_rows(int width, int threads_per_block) {
    const int rows_touched_by_block = (threads_per_block + width - 1) / width + 1;
    return rows_touched_by_block + 2 * MAX_RADIUS;
}

__global__ void bilateral_filter_multi_cuda_kernel(
    const float *src_all,
    float *dst_all,
    int width,
    int height,
    int image_size,
    int window_elems
) {
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pattern_idx = blockIdx.y;

    if (pixel_idx >= image_size) return;

    int x = pixel_idx % width;
    int y = pixel_idx / width;

    size_t base = (size_t)pattern_idx * image_size;
    float center = src_all[base + pixel_idx];

    float sum_w = 0.0f;
    float sum_v = 0.0f;

    for (int k = 0; k < window_elems; k++) {
        int nx = x + c_dxs[k];
        int ny = y + c_dys[k];

        if (nx < 0) nx = 0;
        if (nx >= width) nx = width - 1;
        if (ny < 0) ny = 0;
        if (ny >= height) ny = height - 1;

        float v = src_all[base + ny * width + nx];
        float diff = v - center;
        float range_w = 1.0f / (1.0f + RANGE_ALPHA * diff * diff);
        float w = c_spatial[k] * range_w;

        sum_w += w;
        sum_v += w * v;
    }

    dst_all[base + pixel_idx] = (sum_w > 0.0f) ? (sum_v / sum_w) : center;
}

__global__ void bilateral_filter_multi_shared_cuda_kernel(
    const float *src_all,
    float *dst_all,
    int width,
    int height,
    int image_size,
    int window_elems
) {
    extern __shared__ float tile[];

    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pattern_idx = blockIdx.y;
    int first_idx = blockIdx.x * blockDim.x;
    int last_idx = min(first_idx + blockDim.x - 1, image_size - 1);
    int first_y = first_idx / width;
    int last_y = last_idx / width;
    int tile_origin_y = first_y - MAX_RADIUS;
    int shared_rows = (last_y - first_y + 1) + 2 * MAX_RADIUS;
    int tile_elems = shared_rows * width;
    size_t base = (size_t)pattern_idx * image_size;

    for (int t = threadIdx.x; t < tile_elems; t += blockDim.x) {
        int local_y = t / width;
        int x = t - local_y * width;
        int gy = tile_origin_y + local_y;
        if (gy < 0) gy = 0;
        if (gy >= height) gy = height - 1;
        tile[t] = src_all[base + gy * width + x];
    }

    __syncthreads();

    if (pixel_idx >= image_size) return;

    int x = pixel_idx % width;
    int y = pixel_idx / width;
    float center = src_all[base + pixel_idx];

    float sum_w = 0.0f;
    float sum_v = 0.0f;

    for (int k = 0; k < window_elems; k++) {
        int nx = x + c_dxs[k];
        int ny = y + c_dys[k];

        if (nx < 0) nx = 0;
        if (nx >= width) nx = width - 1;
        if (ny < 0) ny = 0;
        if (ny >= height) ny = height - 1;

        int local_y = ny - tile_origin_y;
        float v = tile[local_y * width + nx];
        float diff = v - center;
        float range_w = 1.0f / (1.0f + RANGE_ALPHA * diff * diff);
        float w = c_spatial[k] * range_w;

        sum_w += w;
        sum_v += w * v;
    }

    dst_all[base + pixel_idx] = (sum_w > 0.0f) ? (sum_v / sum_w) : center;
}

static void compute_stats_range(
    const std::vector<float> &img,
    size_t begin,
    size_t count,
    double &sum,
    double &mean,
    double &mean_square,
    uint64_t &checksum
) {
    sum = 0.0;
    double energy = 0.0;

    for (size_t i = 0; i < count; i++) {
        double v = (double)img[begin + i];
        sum += v;
        energy += v * v;
    }

    mean = sum / (double)count;
    mean_square = energy / (double)count;
    checksum = checksum_u8_rounded_range(img, begin, count);
}

static void print_final_summary(
    const char *input_file,
    const char *output_file,
    int width,
    int height,
    int iterations,
    int patterns,
    int threads_per_block,
    int grid_x,
    int grid_y,
    const char *memory_mode,
    size_t shared_memory_bytes,
    float gpu_ms,
    double gpu_sum_all,
    double gpu_mean_all,
    double gpu_mean_square_all,
    uint64_t gpu_checksum_all,
    double gpu_sum_pattern0,
    double gpu_mean_pattern0,
    double gpu_mean_square_pattern0,
    uint64_t gpu_checksum_pattern0
) {
    const int image_size = width * height;
    const int window_elems = MAX_WINDOW;
    const long long total_pixels_all_patterns = (long long)patterns * image_size;
    const long long total_inner_loop_evaluations =
        (long long)patterns * image_size * window_elems * iterations;
    const long long launched_threads_per_iteration = (long long)grid_x * grid_y * threads_per_block;

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("                  Execution Configuration\n");
    std::printf("============================================================\n");
    std::printf("implementation               = CUDA multi-pattern 2D-grid parallelism\n");
    std::printf("input_file                   = %s\n", input_file);
    std::printf("output_file_pattern0         = %s\n", output_file);
    std::printf("image_width                  = %d\n", width);
    std::printf("image_height                 = %d\n", height);
    std::printf("pixels_per_pattern           = %d\n", image_size);
    std::printf("iterations                   = %d\n", iterations);

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("                   Algorithm Parameters\n");
    std::printf("============================================================\n");
    std::printf("algorithm                    = Bilateral Filter\n");
    std::printf("radius                       = %d\n", MAX_RADIUS);
    std::printf("window_size                  = %d x %d\n", 2 * MAX_RADIUS + 1, 2 * MAX_RADIUS + 1);
    std::printf("window_elems_per_pixel       = %d\n", window_elems);
    std::printf("spatial_alpha                = %.4f\n", SPATIAL_ALPHA);
    std::printf("range_alpha                  = %.4f\n", RANGE_ALPHA);
    std::printf("boundary_policy              = clamp-to-edge\n");
    std::printf("range_kernel                 = 1 / (1 + RANGE_ALPHA * diff * diff)\n");

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("                   Pattern Configuration\n");
    std::printf("============================================================\n");
    std::printf("patterns                     = %d\n", patterns);
    std::printf("pattern0_source              = original input image\n");
    std::printf("pattern1_to_N_source         = deterministic variants of original image\n");
    std::printf("randomness_used              = no\n");
    std::printf("total_pixels_all_patterns    = %lld\n", total_pixels_all_patterns);

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("                       CUDA Mapping\n");
    std::printf("============================================================\n");
    std::printf("part                         = Part 5\n");
    std::printf("execution_model              = CUDA SIMT with 2D grid\n");
    std::printf("thread_mapping               = one CUDA thread = one output pixel of one pattern\n");
    std::printf("grid_x_mapping               = output pixel index blocks\n");
    std::printf("grid_y_mapping               = pattern index via blockIdx.y\n");
    std::printf("threads_per_block            = %d\n", threads_per_block);
    std::printf("cuda_grid_x                  = %d\n", grid_x);
    std::printf("cuda_grid_y                  = %d\n", grid_y);
    std::printf("memory_mode                  = %s\n", memory_mode);
    std::printf("shared_memory_bytes_per_blk  = %zu\n", shared_memory_bytes);
    std::printf("active_threads_per_iter      = %lld\n", total_pixels_all_patterns);
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
    std::printf("gpu_multi_pattern_kernel_ms  = %.6f\n", gpu_ms);

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("                    Verification Output\n");
    std::printf("============================================================\n");
    std::printf("gpu_output_sum_all_patterns  = %.6f\n", gpu_sum_all);
    std::printf("gpu_output_mean_all_patterns = %.6f\n", gpu_mean_all);
    std::printf("gpu_output_mean_square_all   = %.6f\n", gpu_mean_square_all);
    std::printf("gpu_output_checksum_all      = 0x%016llx\n", (unsigned long long)gpu_checksum_all);
    std::printf("gpu_output_sum_pattern0      = %.6f\n", gpu_sum_pattern0);
    std::printf("gpu_output_mean_pattern0     = %.6f\n", gpu_mean_pattern0);
    std::printf("gpu_output_mean_square_p0    = %.6f\n", gpu_mean_square_pattern0);
    std::printf("gpu_output_checksum_pattern0 = 0x%016llx\n", (unsigned long long)gpu_checksum_pattern0);
    std::printf("compare_with_cpu_output      = run ./main_cpu and compare checksum/output_sum\n");

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("                       Output Status\n");
    std::printf("============================================================\n");
    std::printf("output_written_pattern0      = %s\n", output_file);

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("                      Program Finished\n");
    std::printf("============================================================\n");
    std::printf("\n");
}

int main(int argc, char **argv) {
    const char *input_file  = (argc >= 2) ? argv[1] : DEFAULT_INPUT_FILE;
    const char *output_file = (argc >= 3) ? argv[2] : DEFAULT_OUTPUT_FILE;
    int iterations          = (argc >= 4) ? std::atoi(argv[3]) : DEFAULT_ITERATIONS;
    int patterns            = (argc >= 5) ? std::atoi(argv[4]) : DEFAULT_PATTERNS;
    int threads_per_block   = (argc >= 6) ? std::atoi(argv[5]) : DEFAULT_THREADS_PER_BLOCK;
    const char *memory_mode = (argc >= 7) ? argv[6] : "shared";
    bool use_shared_memory = true;

    if (iterations <= 0) iterations = DEFAULT_ITERATIONS;
    if (patterns <= 0) patterns = DEFAULT_PATTERNS;
    if (threads_per_block <= 0 || threads_per_block > 1024) threads_per_block = DEFAULT_THREADS_PER_BLOCK;
    if (std::strcmp(memory_mode, "shared") == 0) {
        use_shared_memory = true;
    } else if (std::strcmp(memory_mode, "global") == 0 || std::strcmp(memory_mode, "naive") == 0) {
        use_shared_memory = false;
        memory_mode = "global";
    } else {
        std::fprintf(stderr, "ERROR: memory mode must be shared or global.\n");
        return 1;
    }

    int width = 0;
    int height = 0;
    std::vector<float> base_image;

    if (!read_txt_image(input_file, width, height, base_image)) {
        return 1;
    }

    std::vector<float> spatial;
    std::vector<int> dxs;
    std::vector<int> dys;
    prepare_spatial_kernel(MAX_RADIUS, spatial, dxs, dys);

    const int window_elems = (int)spatial.size();
    const int image_size = width * height;

    std::vector<float> input_all;
    generate_patterns_from_base(base_image, width, height, patterns, input_all);

    cudaCheck(cudaMemcpyToSymbol(c_spatial, spatial.data(), window_elems * sizeof(float)), "cudaMemcpyToSymbol c_spatial");
    cudaCheck(cudaMemcpyToSymbol(c_dxs, dxs.data(), window_elems * sizeof(int)), "cudaMemcpyToSymbol c_dxs");
    cudaCheck(cudaMemcpyToSymbol(c_dys, dys.data(), window_elems * sizeof(int)), "cudaMemcpyToSymbol c_dys");

    const size_t total_count = (size_t)patterns * image_size;
    const size_t total_bytes = total_count * sizeof(float);

    float *d_a = nullptr;
    float *d_b = nullptr;

    cudaCheck(cudaMalloc((void **)&d_a, total_bytes), "cudaMalloc d_a");
    cudaCheck(cudaMalloc((void **)&d_b, total_bytes), "cudaMalloc d_b");
    cudaCheck(cudaMemcpy(d_a, input_all.data(), total_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    const int grid_x = (image_size + threads_per_block - 1) / threads_per_block;
    const int grid_y = patterns;
    dim3 block(threads_per_block);
    dim3 grid(grid_x, grid_y);
    size_t shared_memory_bytes = 0;
    if (use_shared_memory) {
        const int shared_rows = compute_shared_rows(width, threads_per_block);
        shared_memory_bytes = (size_t)width * shared_rows * sizeof(float);

        int device = 0;
        cudaDeviceProp prop;
        cudaCheck(cudaGetDevice(&device), "cudaGetDevice");
        cudaCheck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
        if (shared_memory_bytes > (size_t)prop.sharedMemPerBlock) {
            std::printf("WARNING: requested shared memory %zu bytes exceeds device limit %zu bytes; using global mode.\n",
                        shared_memory_bytes, (size_t)prop.sharedMemPerBlock);
            use_shared_memory = false;
            memory_mode = "global";
            shared_memory_bytes = 0;
        }
    }

    cudaEvent_t ev_st;
    cudaEvent_t ev_ed;
    cudaCheck(cudaEventCreate(&ev_st), "cudaEventCreate start");
    cudaCheck(cudaEventCreate(&ev_ed), "cudaEventCreate end");

    cudaCheck(cudaEventRecord(ev_st), "cudaEventRecord start");

    for (int it = 0; it < iterations; it++) {
        if (use_shared_memory) {
            bilateral_filter_multi_shared_cuda_kernel<<<grid, block, shared_memory_bytes>>>(d_a, d_b, width, height, image_size, window_elems);
        } else {
            bilateral_filter_multi_cuda_kernel<<<grid, block>>>(d_a, d_b, width, height, image_size, window_elems);
        }
        cudaCheck(cudaGetLastError(), "kernel launch");
        std::swap(d_a, d_b);
    }

    cudaCheck(cudaEventRecord(ev_ed), "cudaEventRecord end");
    cudaCheck(cudaEventSynchronize(ev_ed), "cudaEventSynchronize end");

    float gpu_ms = 0.0f;
    cudaCheck(cudaEventElapsedTime(&gpu_ms, ev_st, ev_ed), "cudaEventElapsedTime");

    std::vector<float> gpu_out_all(total_count);
    cudaCheck(cudaMemcpy(gpu_out_all.data(), d_a, total_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    std::vector<float> gpu_pattern0(gpu_out_all.begin(), gpu_out_all.begin() + image_size);
    if (!write_txt_image(output_file, width, height, gpu_pattern0)) {
        std::fprintf(stderr, "Failed to write output file: %s\n", output_file);
    }

    double gpu_sum_all, gpu_mean_all, gpu_mean_square_all;
    double gpu_sum_pattern0, gpu_mean_pattern0, gpu_mean_square_pattern0;
    uint64_t gpu_checksum_all, gpu_checksum_pattern0;

    compute_stats_range(gpu_out_all, 0, gpu_out_all.size(),
                        gpu_sum_all, gpu_mean_all, gpu_mean_square_all, gpu_checksum_all);
    compute_stats_range(gpu_out_all, 0, image_size,
                        gpu_sum_pattern0, gpu_mean_pattern0, gpu_mean_square_pattern0, gpu_checksum_pattern0);

    print_final_summary(input_file, output_file, width, height, iterations,
                        patterns, threads_per_block, grid_x, grid_y,
                        memory_mode, shared_memory_bytes,
                        gpu_ms,
                        gpu_sum_all, gpu_mean_all, gpu_mean_square_all,
                        gpu_checksum_all,
                        gpu_sum_pattern0, gpu_mean_pattern0, gpu_mean_square_pattern0,
                        gpu_checksum_pattern0);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaEventDestroy(ev_st);
    cudaEventDestroy(ev_ed);

    return 0;
}
