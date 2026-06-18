#include <stdio.h>
#include <stdlib.h>

// -----------------------------------------------------------------------------
// Part 2: RVV Vector Reduction implementation for Bilateral Filter
//
// Default execution:
//   ./main
// Optional execution:
//   ./main <input_txt> <output_txt> [iterations]
//
// RVV mapping:
//   one output pixel at a time; vector lanes process the 9x9 window and reduce.
// -----------------------------------------------------------------------------

#define DEFAULT_INPUT_FILE  "../test/cyberpunk2077_in.txt"
#define DEFAULT_OUTPUT_FILE "output/rvv_out.txt"

#define RADIUS 4
#define WINDOW_SIZE (2 * RADIUS + 1)
#define KERNEL_ELEMS (WINDOW_SIZE * WINDOW_SIZE)
#define DEFAULT_ITERATIONS 3

#define SPATIAL_ALPHA 0.0600f
#define RANGE_ALPHA   0.0009f
#define MAX_VL_FLOAT  8

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
        printf("ERROR: malloc failed for input image\n");
        fclose(fp);
        return 0;
    }

    for (int i = 0; i < n; i++) {
        int pixel = 0;
        if (fscanf(fp, "%d", &pixel) != 1) {
            printf("ERROR: not enough pixel values in %s\n", path);
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

static void rvv_reduce_bilateral_window(
    const float *neighbor,
    const float *spatial,
    int count,
    float center,
    float *weighted_sum_out,
    float *weight_sum_out
) {
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    const float one = 1.0f;
    const float alpha = RANGE_ALPHA;

    int offset = 0;
    while (offset < count) {
        const int remaining = count - offset;
        const int vl = (remaining > MAX_VL_FLOAT) ? MAX_VL_FLOAT : remaining;
        float chunk_weighted_sum = 0.0f;
        float chunk_weight_sum = 0.0f;

        const float *neighbor_ptr = neighbor + offset;
        const float *spatial_ptr = spatial + offset;

        asm volatile(
            // Configure vector length for FP32 elements.
            "mv t0, %[vl]\n\t"
            "vsetvli zero, t0, e32, m1, ta, ma\n\t"

            // v1 = neighbor values, v2 = spatial weights.
            "vle32.v v1, (%[neighbor_ptr])\n\t"
            "vle32.v v2, (%[spatial_ptr])\n\t"

            // v3 = diff = neighbor - center.
            "vfsub.vf v3, v1, %[center]\n\t"

            // v3 = diff * diff.
            "vfmul.vv v3, v3, v3\n\t"

            // v3 = 1.0 + alpha * diff * diff.
            "vfmul.vf v3, v3, %[alpha]\n\t"
            "vfadd.vf v3, v3, %[one]\n\t"

            // v4 = range_weight = 1.0 / v3.
            "vfmv.v.f v4, %[one]\n\t"
            "vfdiv.vv v4, v4, v3\n\t"

            // v5 = weight = spatial_weight * range_weight.
            "vfmul.vv v5, v2, v4\n\t"

            // v6 = weighted value = weight * neighbor.
            "vfmul.vv v6, v5, v1\n\t"

            // v7[0] = sum(v6), v8[0] = sum(v5).
            "vmv.v.i v0, 0\n\t"
            "vfredusum.vs v7, v6, v0\n\t"
            "vfredusum.vs v8, v5, v0\n\t"

            // Store the scalar reduction results back to C variables.
            "vfmv.f.s ft0, v7\n\t"
            "fsw ft0, 0(%[chunk_weighted_sum_ptr])\n\t"
            "vfmv.f.s ft1, v8\n\t"
            "fsw ft1, 0(%[chunk_weight_sum_ptr])\n\t"
            :
            : [neighbor_ptr] "r"(neighbor_ptr),
              [spatial_ptr] "r"(spatial_ptr),
              [vl] "r"(vl),
              [center] "f"(center),
              [alpha] "f"(alpha),
              [one] "f"(one),
              [chunk_weighted_sum_ptr] "r"(&chunk_weighted_sum),
              [chunk_weight_sum_ptr] "r"(&chunk_weight_sum)
            : "t0", "ft0", "ft1", "memory"
        );

        weighted_sum += chunk_weighted_sum;
        weight_sum += chunk_weight_sum;
        offset += vl;
    }

    *weighted_sum_out = weighted_sum;
    *weight_sum_out = weight_sum;
}

static void bilateral_filter_rvv(
    const float *src,
    float *dst,
    int width,
    int height,
    const float spatial[KERNEL_ELEMS],
    const int dxs[KERNEL_ELEMS],
    const int dys[KERNEL_ELEMS]
) {
    float neighbor_values[KERNEL_ELEMS];

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const int center_idx = y * width + x;
            const float center = src[center_idx];

            // Gather the 9x9 window into a contiguous temporary array so that
            // RVV can use unit-stride vector loads in Part 2.
            for (int k = 0; k < KERNEL_ELEMS; k++) {
                const int nx = clamp_int(x + dxs[k], 0, width - 1);
                const int ny = clamp_int(y + dys[k], 0, height - 1);
                neighbor_values[k] = src[ny * width + nx];
            }

            float weighted_sum = 0.0f;
            float weight_sum = 0.0f;
            rvv_reduce_bilateral_window(neighbor_values, spatial, KERNEL_ELEMS,
                                        center, &weighted_sum, &weight_sum);
            dst[center_idx] = weighted_sum / weight_sum;
        }
    }
}

static void run_bilateral_iterations(
    float *input,
    float *buffer_a,
    float *buffer_b,
    int width,
    int height,
    int iterations,
    const float spatial[KERNEL_ELEMS],
    const int dxs[KERNEL_ELEMS],
    const int dys[KERNEL_ELEMS],
    float **final_output
) {
    const int n = width * height;

    for (int i = 0; i < n; i++) {
        buffer_a[i] = input[i];
        buffer_b[i] = 0.0f;
    }

    float *src = buffer_a;
    float *dst = buffer_b;

    for (int iter = 0; iter < iterations; iter++) {
        bilateral_filter_rvv(src, dst, width, height, spatial, dxs, dys);
        float *tmp = src;
        src = dst;
        dst = tmp;
    }

    *final_output = src;
}

static float verify_constant_image(
    const float spatial[KERNEL_ELEMS],
    const int dxs[KERNEL_ELEMS],
    const int dys[KERNEL_ELEMS]
) {
    const int width = 16;
    const int height = 16;
    const int n = width * height;
    float *src = (float *)malloc(sizeof(float) * n);
    float *dst = (float *)malloc(sizeof(float) * n);

    if (!src || !dst) {
        printf("ERROR: malloc failed in verify_constant_image\n");
        free(src);
        free(dst);
        return -1.0f;
    }

    for (int i = 0; i < n; i++) {
        src[i] = 128.0f;
        dst[i] = 0.0f;
    }

    bilateral_filter_rvv(src, dst, width, height, spatial, dxs, dys);

    float max_abs_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = dst[i] - 128.0f;
        if (err < 0.0f) err = -err;
        if (err > max_abs_err) max_abs_err = err;
    }

    free(src);
    free(dst);
    return max_abs_err;
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

static void print_summary(
    const char *input_file,
    const char *output_file,
    const float *image,
    int width,
    int height,
    int iterations,
    float constant_image_max_abs_err
) {
    const int n = width * height;
    double sum = 0.0;
    double energy = 0.0;

    for (int i = 0; i < n; i++) {
        sum += (double)image[i];
        energy += (double)image[i] * (double)image[i];
    }

    const double mean = sum / (double)n;
    const double mean_square = energy / (double)n;
    const unsigned long long checksum = checksum_u8_from_float(image, n);
    const long long total_inner_loop_evaluations = (long long)n * KERNEL_ELEMS * iterations;
    const int vector_chunks_per_pixel = (KERNEL_ELEMS + MAX_VL_FLOAT - 1) / MAX_VL_FLOAT;

    printf("\n");
    printf("============================================================\n");
    printf("                  Execution Configuration\n");
    printf("============================================================\n");
    printf("implementation               = RVV vector reduction\n");
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
    printf("                    Parallelism Mapping\n");
    printf("============================================================\n");
    printf("part                         = Part 2\n");
    printf("execution_model              = RISC-V Vector Extension (RVV)\n");
    printf("rvv_mapping                  = one output pixel, vectorize 9x9 window summation\n");
    printf("rvv_max_vl_float             = %d\n", MAX_VL_FLOAT);
    printf("vector_chunks_per_pixel      = %d\n", vector_chunks_per_pixel);
    printf("unit_stride_vector_load      = yes (vle32.v)\n");
    printf("vector_reduction_used        = yes (vfredusum.vs)\n");
    printf("strided_vector_access_used   = no\n");
    printf("cuda_used                    = no\n");

    printf("\n");
    printf("============================================================\n");
    printf("                      Workload Summary\n");
    printf("============================================================\n");
    printf("total_inner_loop_evaluations = %lld\n", total_inner_loop_evaluations);
    printf("operations_per_neighbor      = range weight + weighted sum + weight sum\n");
    printf("preprocessing_in_timing      = no\n");

    printf("\n");
    printf("============================================================\n");
    printf("                    Verification Output\n");
    printf("============================================================\n");
    printf("constant_image_max_abs_err   = %.8f\n", constant_image_max_abs_err);
    printf("output_sum                   = %.6f\n", sum);
    printf("output_mean                  = %.6f\n", mean);
    printf("output_mean_square           = %.6f\n", mean_square);
    printf("output_checksum              = 0x%016llx\n", checksum);
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

int main(int argc, char **argv) {
    const char *input_file = DEFAULT_INPUT_FILE;
    const char *output_file = DEFAULT_OUTPUT_FILE;
    int iterations = DEFAULT_ITERATIONS;

    if (argc >= 2) input_file = argv[1];
    if (argc >= 3) output_file = argv[2];
    if (argc >= 4) iterations = atoi(argv[3]);

    if (iterations <= 0) {
        printf("ERROR: iterations must be positive.\n");
        return 1;
    }

    float *input = NULL;
    int width = 0;
    int height = 0;

    if (!load_image_txt(input_file, &input, &width, &height)) {
        return 1;
    }

    const int n = width * height;
    float *buffer_a = (float *)malloc(sizeof(float) * n);
    float *buffer_b = (float *)malloc(sizeof(float) * n);
    if (!buffer_a || !buffer_b) {
        printf("ERROR: malloc failed for working buffers\n");
        free(input);
        free(buffer_a);
        free(buffer_b);
        return 1;
    }

    float spatial[KERNEL_ELEMS];
    int dxs[KERNEL_ELEMS];
    int dys[KERNEL_ELEMS];
    precompute_spatial_kernel(spatial, dxs, dys);

    // Small correctness check. A constant image should remain constant after filtering.
    float constant_image_max_abs_err = verify_constant_image(spatial, dxs, dys);

    float *final_output = NULL;
    run_bilateral_iterations(input, buffer_a, buffer_b, width, height,
                             iterations, spatial, dxs, dys, &final_output);

    print_summary(input_file, output_file, final_output, width, height,
                  iterations, constant_image_max_abs_err);
    write_output_txt(output_file, final_output, width, height);

    free(input);
    free(buffer_a);
    free(buffer_b);
    return 0;
}
