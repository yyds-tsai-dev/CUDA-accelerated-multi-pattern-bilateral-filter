#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>
#include <algorithm>

// -----------------------------------------------------------------------------
// Part 5 CPU multi-pattern baseline for Bilateral Filter
//
// This standalone CPU program is paired with p5/main.cu.  It uses the same input
// pattern generation and filter formula as the CUDA multi-pattern version.
// -----------------------------------------------------------------------------

#define MAX_RADIUS 4
#define MAX_WINDOW ((2 * MAX_RADIUS + 1) * (2 * MAX_RADIUS + 1))
#define DEFAULT_INPUT_FILE  "../test/cyberpunk2077_in.txt"
#define DEFAULT_OUTPUT_FILE "output/cpu_multi_out.txt"
#define DEFAULT_ITERATIONS 3
#define DEFAULT_PATTERNS 32

#define SPATIAL_ALPHA 0.0600f
#define RANGE_ALPHA   0.0009f

static inline float clampf_host(float x, float lo, float hi) {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
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

static void bilateral_one_pass_cpu(
    const float *src,
    float *dst,
    int width,
    int height,
    int radius,
    const std::vector<float> &spatial,
    const std::vector<int> &dxs,
    const std::vector<int> &dys
) {
    const int window_elems = (int)spatial.size();

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float center = src[y * width + x];
            float sum_w = 0.0f;
            float sum_v = 0.0f;

            for (int k = 0; k < window_elems; k++) {
                int nx = x + dxs[k];
                int ny = y + dys[k];

                if (nx < 0) nx = 0;
                if (nx >= width) nx = width - 1;
                if (ny < 0) ny = 0;
                if (ny >= height) ny = height - 1;

                float v = src[ny * width + nx];
                float diff = v - center;
                float range_w = 1.0f / (1.0f + RANGE_ALPHA * diff * diff);
                float w = spatial[k] * range_w;

                sum_w += w;
                sum_v += w * v;
            }

            dst[y * width + x] = (sum_w > 0.0f) ? (sum_v / sum_w) : center;
        }
    }
}

static void bilateral_multi_cpu(
    const std::vector<float> &input_all,
    std::vector<float> &output_all,
    int width,
    int height,
    int patterns,
    int radius,
    int iterations,
    const std::vector<float> &spatial,
    const std::vector<int> &dxs,
    const std::vector<int> &dys
) {
    const int image_size = width * height;
    std::vector<float> buf_a = input_all;
    std::vector<float> buf_b((size_t)patterns * image_size, 0.0f);

    float *cur = buf_a.data();
    float *nxt = buf_b.data();

    for (int it = 0; it < iterations; it++) {
        for (int p = 0; p < patterns; p++) {
            const float *src = cur + (size_t)p * image_size;
            float *dst = nxt + (size_t)p * image_size;
            bilateral_one_pass_cpu(src, dst, width, height, radius, spatial, dxs, dys);
        }
        std::swap(cur, nxt);
    }

    output_all.assign(cur, cur + (size_t)patterns * image_size);
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
    double cpu_ms,
    double cpu_sum_all,
    double cpu_mean_all,
    double cpu_mean_square_all,
    uint64_t cpu_checksum_all,
    double cpu_sum_pattern0,
    double cpu_mean_pattern0,
    double cpu_mean_square_pattern0,
    uint64_t cpu_checksum_pattern0
) {
    const int image_size = width * height;
    const int window_elems = MAX_WINDOW;
    const long long total_pixels_all_patterns = (long long)patterns * image_size;
    const long long total_inner_loop_evaluations =
        (long long)patterns * image_size * window_elems * iterations;

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("                  Execution Configuration\n");
    std::printf("============================================================\n");
    std::printf("implementation               = CPU multi-pattern Bilateral Filter\n");
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
    std::printf("                      Workload Summary\n");
    std::printf("============================================================\n");
    std::printf("total_inner_loop_evaluations = %lld\n", total_inner_loop_evaluations);
    std::printf("operations_per_neighbor      = range weight + weighted sum + weight sum\n");

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("                       Runtime Summary\n");
    std::printf("============================================================\n");
    std::printf("cpu_multi_pattern_time_ms    = %.6f\n", cpu_ms);
    std::printf("timing_method_cpu            = native CPU chrono timing\n");
    std::printf("file_io_in_timing            = no\n");

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("                    Verification Output\n");
    std::printf("============================================================\n");
    std::printf("cpu_output_sum_all_patterns  = %.6f\n", cpu_sum_all);
    std::printf("cpu_output_mean_all_patterns = %.6f\n", cpu_mean_all);
    std::printf("cpu_output_mean_square_all   = %.6f\n", cpu_mean_square_all);
    std::printf("cpu_output_checksum_all      = 0x%016llx\n", (unsigned long long)cpu_checksum_all);
    std::printf("cpu_output_sum_pattern0      = %.6f\n", cpu_sum_pattern0);
    std::printf("cpu_output_mean_pattern0     = %.6f\n", cpu_mean_pattern0);
    std::printf("cpu_output_mean_square_p0    = %.6f\n", cpu_mean_square_pattern0);
    std::printf("cpu_output_checksum_pattern0 = 0x%016llx\n", (unsigned long long)cpu_checksum_pattern0);
    std::printf("compare_with_gpu_output      = run ./main and compare checksum/output_sum\n");

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

    if (iterations <= 0) iterations = DEFAULT_ITERATIONS;
    if (patterns <= 0) patterns = DEFAULT_PATTERNS;

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

    const int image_size = width * height;

    std::vector<float> input_all;
    generate_patterns_from_base(base_image, width, height, patterns, input_all);

    std::vector<float> cpu_out_all;
    auto cpu_st = std::chrono::high_resolution_clock::now();
    bilateral_multi_cpu(input_all, cpu_out_all, width, height, patterns, MAX_RADIUS,
                        iterations, spatial, dxs, dys);
    auto cpu_ed = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_elapsed = cpu_ed - cpu_st;
    double cpu_ms = cpu_elapsed.count() * 1000.0;

    std::vector<float> cpu_pattern0(cpu_out_all.begin(), cpu_out_all.begin() + image_size);
    if (!write_txt_image(output_file, width, height, cpu_pattern0)) {
        std::fprintf(stderr, "Failed to write output file: %s\n", output_file);
    }

    double cpu_sum_all, cpu_mean_all, cpu_mean_square_all;
    double cpu_sum_pattern0, cpu_mean_pattern0, cpu_mean_square_pattern0;
    uint64_t cpu_checksum_all, cpu_checksum_pattern0;

    compute_stats_range(cpu_out_all, 0, cpu_out_all.size(),
                        cpu_sum_all, cpu_mean_all, cpu_mean_square_all, cpu_checksum_all);
    compute_stats_range(cpu_out_all, 0, image_size,
                        cpu_sum_pattern0, cpu_mean_pattern0, cpu_mean_square_pattern0, cpu_checksum_pattern0);

    print_final_summary(input_file, output_file, width, height, iterations,
                        patterns, cpu_ms,
                        cpu_sum_all, cpu_mean_all, cpu_mean_square_all,
                        cpu_checksum_all,
                        cpu_sum_pattern0, cpu_mean_pattern0, cpu_mean_square_pattern0,
                        cpu_checksum_pattern0);

    return 0;
}
