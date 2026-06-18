#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// -----------------------------------------------------------------------------
// Part 4 CPU baseline for Bilateral Filter
//
// This standalone CPU program is paired with p4/main.cu.  It uses the same input,
// same filter formula, and same output format, but runs on native CPU using C++.
// -----------------------------------------------------------------------------

#define DEFAULT_INPUT_FILE  "../test/cyberpunk2077_in.txt"
#define DEFAULT_OUTPUT_FILE "output/cpu_native_out.txt"

#define RADIUS 4
#define WINDOW_SIZE (2 * RADIUS + 1)
#define KERNEL_ELEMS (WINDOW_SIZE * WINDOW_SIZE)
#define DEFAULT_ITERATIONS 3

#define SPATIAL_ALPHA 0.0600f
#define RANGE_ALPHA   0.0009f

static inline int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static int load_image_txt(const char *path, float **image, int *width, int *height) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        printf("ERROR: cannot open input file %s\n", path);
        return 0;
    }

    if (fscanf(fp, "%d %d", width, height) != 2) {
        printf("ERROR: invalid input header in %s\n", path);
        fclose(fp);
        return 0;
    }

    const int n = (*width) * (*height);
    *image = (float *)malloc(sizeof(float) * n);
    if (!(*image)) {
        printf("ERROR: malloc failed\n");
        fclose(fp);
        return 0;
    }

    for (int i = 0; i < n; i++) {
        int pixel = 0;
        if (fscanf(fp, "%d", &pixel) != 1) {
            printf("ERROR: not enough pixels in %s\n", path);
            free(*image);
            *image = NULL;
            fclose(fp);
            return 0;
        }
        if (pixel < 0) pixel = 0;
        if (pixel > 255) pixel = 255;
        (*image)[i] = (float)pixel;
    }

    fclose(fp);
    return 1;
}

static void write_output_txt(const char *path, const float *image, int width, int height) {
    FILE *fp = fopen(path, "w");
    if (!fp) {
        printf("WARNING: cannot write output file %s\n", path);
        return;
    }

    fprintf(fp, "%d %d\n", width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pixel = (int)(image[y * width + x] + 0.5f);
            if (pixel < 0) pixel = 0;
            if (pixel > 255) pixel = 255;
            fprintf(fp, "%d%c", pixel, (x == width - 1) ? '\n' : ' ');
        }
    }

    fclose(fp);
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

static void bilateral_filter_cpu(
    const float *src,
    float *dst,
    int width,
    int height,
    const float spatial[KERNEL_ELEMS],
    const int dxs[KERNEL_ELEMS],
    const int dys[KERNEL_ELEMS]
) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const int center_idx = y * width + x;
            const float center = src[center_idx];
            float weighted_sum = 0.0f;
            float weight_sum = 0.0f;

            for (int k = 0; k < KERNEL_ELEMS; k++) {
                const int nx = clamp_int(x + dxs[k], 0, width - 1);
                const int ny = clamp_int(y + dys[k], 0, height - 1);
                const float neighbor = src[ny * width + nx];
                const float diff = neighbor - center;
                const float range_weight = 1.0f / (1.0f + RANGE_ALPHA * diff * diff);
                const float weight = spatial[k] * range_weight;
                weighted_sum += weight * neighbor;
                weight_sum += weight;
            }

            dst[center_idx] = weighted_sum / weight_sum;
        }
    }
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
    double cpu_ms,
    double sum,
    double mean,
    double mean_square,
    unsigned long long checksum
) {
    const int n = width * height;
    const long long total_inner_loop_evaluations = (long long)n * KERNEL_ELEMS * iterations;

    printf("\n");
    printf("============================================================\n");
    printf("                  Execution Configuration\n");
    printf("============================================================\n");
    printf("implementation               = CPU native Bilateral Filter\n");
    printf("input_file                   = %s\n", input_file);
    printf("output_file                  = %s\n", output_file);
    printf("image_width                  = %d\n", width);
    printf("image_height                 = %d\n", height);
    printf("total_pixels                 = %d\n", n);
    printf("iterations                   = %d\n", iterations);

    printf("\n");
    printf("============================================================\n");
    printf("                   Algorithm Parameters\n");
    printf("============================================================\n");
    printf("algorithm                    = Bilateral Filter\n");
    printf("radius                       = %d\n", RADIUS);
    printf("window_size                  = %d x %d\n", WINDOW_SIZE, WINDOW_SIZE);
    printf("window_elems_per_pixel       = %d\n", KERNEL_ELEMS);
    printf("spatial_alpha                = %.4f\n", SPATIAL_ALPHA);
    printf("range_alpha                  = %.4f\n", RANGE_ALPHA);
    printf("boundary_policy              = clamp-to-edge\n");
    printf("range_kernel                 = 1 / (1 + RANGE_ALPHA * diff * diff)\n");

    printf("\n");
    printf("============================================================\n");
    printf("                      Workload Summary\n");
    printf("============================================================\n");
    printf("total_inner_loop_evaluations = %lld\n", total_inner_loop_evaluations);
    printf("operations_per_neighbor      = range weight + weighted sum + weight sum\n");

    printf("\n");
    printf("============================================================\n");
    printf("                       Runtime Summary\n");
    printf("============================================================\n");
    printf("cpu_native_time_ms           = %.6f\n", cpu_ms);
    printf("timing_method_cpu            = native CPU chrono timing\n");
    printf("file_io_in_timing            = no\n");

    printf("\n");
    printf("============================================================\n");
    printf("                    Verification Output\n");
    printf("============================================================\n");
    printf("cpu_output_sum               = %.6f\n", sum);
    printf("cpu_output_mean              = %.6f\n", mean);
    printf("cpu_output_mean_square       = %.6f\n", mean_square);
    printf("cpu_output_checksum          = 0x%016llx\n", checksum);
    printf("compare_with_gpu_output      = run ./main and compare checksum/output_sum\n");

    printf("\n");
    printf("============================================================\n");
    printf("                       Output Status\n");
    printf("============================================================\n");
    printf("output_written               = %s\n", output_file);

    printf("\n");
    printf("============================================================\n");
    printf("                      Program Finished\n");
    printf("============================================================\n");
    printf("\n");
}

int main(int argc, char **argv) {
    const char *input_file = (argc >= 2) ? argv[1] : DEFAULT_INPUT_FILE;
    const char *output_file = (argc >= 3) ? argv[2] : DEFAULT_OUTPUT_FILE;
    int iterations = (argc >= 4) ? atoi(argv[3]) : DEFAULT_ITERATIONS;
    if (iterations <= 0) iterations = DEFAULT_ITERATIONS;

    float *input = NULL;
    int width = 0;
    int height = 0;
    if (!load_image_txt(input_file, &input, &width, &height)) return 1;

    const int n = width * height;
    float *a = (float *)malloc(sizeof(float) * n);
    float *b = (float *)malloc(sizeof(float) * n);
    if (!a || !b) {
        printf("ERROR: malloc failed\n");
        free(input);
        free(a);
        free(b);
        return 1;
    }

    for (int i = 0; i < n; i++) {
        a[i] = input[i];
        b[i] = 0.0f;
    }

    float spatial[KERNEL_ELEMS];
    int dxs[KERNEL_ELEMS];
    int dys[KERNEL_ELEMS];
    precompute_spatial_kernel(spatial, dxs, dys);

    float *src = a;
    float *dst = b;
    auto st = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        bilateral_filter_cpu(src, dst, width, height, spatial, dxs, dys);
        float *tmp = src;
        src = dst;
        dst = tmp;
    }
    auto ed = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = ed - st;
    const double cpu_ms = elapsed.count() * 1000.0;

    double sum, mean, mean_square;
    unsigned long long checksum;
    compute_image_stats(src, n, &sum, &mean, &mean_square, &checksum);

    write_output_txt(output_file, src, width, height);
    print_final_summary(input_file, output_file, width, height, iterations,
                        cpu_ms, sum, mean, mean_square, checksum);

    free(input);
    free(a);
    free(b);
    return 0;
}
