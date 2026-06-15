#include "../common/bilateral_common.h"
#include "../common/result_io.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#define CUDA_CHECK(call)                                                                     \
  do {                                                                                       \
    cudaError_t status = (call);                                                             \
    if (status != cudaSuccess) {                                                             \
      std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" \
                << __LINE__ << "\n";                                                        \
      std::exit(EXIT_FAILURE);                                                               \
    }                                                                                        \
  } while (0)

__device__ int d_clamp_int(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

__device__ float d_spatial_weight(int dx, int dy, float sigma_s2) {
  const float d2 = static_cast<float>(dx * dx + dy * dy);
  return 1.0f / (1.0f + d2 / sigma_s2);
}

__device__ float d_range_weight(float center, float neighbor, float sigma_r2) {
  const float diff = center - neighbor;
  return 1.0f / (1.0f + (diff * diff) / sigma_r2);
}

__global__ void bilateral_multi_pattern_kernel(const float* input,
                                               float* output,
                                               BilateralParams params,
                                               int patterns) {
  const int pattern = blockIdx.y;
  const size_t image_size = static_cast<size_t>(params.width) * static_cast<size_t>(params.height);
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= image_size || pattern >= patterns) {
    return;
  }

  const int y = static_cast<int>(index / static_cast<size_t>(params.width));
  const int x = static_cast<int>(index - static_cast<size_t>(y) * params.width);
  const size_t base = static_cast<size_t>(pattern) * image_size;
  const float center = input[base + index];
  float numerator = 0.0f;
  float denominator = 0.0f;

  for (int dy = -params.radius; dy <= params.radius; ++dy) {
    const int yy = d_clamp_int(y + dy, 0, params.height - 1);
    for (int dx = -params.radius; dx <= params.radius; ++dx) {
      const int xx = d_clamp_int(x + dx, 0, params.width - 1);
      const size_t neighbor_index = static_cast<size_t>(yy) * params.width + xx;
      const float neighbor = input[base + neighbor_index];
      const float weight = d_spatial_weight(dx, dy, params.sigma_s2) *
                           d_range_weight(center, neighbor, params.sigma_r2);
      numerator += neighbor * weight;
      denominator += weight;
    }
  }

  output[base + index] = numerator / denominator;
}

static bool valid_allocation_size(int width, int height, int patterns) {
  const size_t image_size = static_cast<size_t>(width) * static_cast<size_t>(height);
  const size_t max_size = std::numeric_limits<size_t>::max();
  if (image_size > max_size / static_cast<size_t>(patterns)) {
    return false;
  }

  const size_t count = image_size * static_cast<size_t>(patterns);
  return count <= max_size / sizeof(float);
}

static bool valid_cuda_grid_y(int patterns) {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  return patterns <= prop.maxGridSize[1];
}

static void generate_patterns(std::vector<float>& images, int width, int height, int patterns) {
  const size_t image_size = static_cast<size_t>(width) * static_cast<size_t>(height);
  images.resize(image_size * static_cast<size_t>(patterns));

  std::vector<float> image;
  for (int pattern = 0; pattern < patterns; ++pattern) {
    generate_image(image, width, height, pattern);
    const size_t offset = static_cast<size_t>(pattern) * image_size;
    std::copy(image.begin(), image.end(), images.begin() + static_cast<std::ptrdiff_t>(offset));
  }
}

static void scalar_patterns_reference(const std::vector<float>& input,
                                      std::vector<float>& reference,
                                      const BilateralParams& params,
                                      int patterns) {
  const size_t image_size = static_cast<size_t>(params.width) * static_cast<size_t>(params.height);
  reference.resize(image_size * static_cast<size_t>(patterns));

  std::vector<float> image(image_size);
  std::vector<float> filtered;
  for (int pattern = 0; pattern < patterns; ++pattern) {
    const size_t offset = static_cast<size_t>(pattern) * image_size;
    std::copy(input.begin() + static_cast<std::ptrdiff_t>(offset),
              input.begin() + static_cast<std::ptrdiff_t>(offset + image_size),
              image.begin());
    scalar_bilateral_filter(image, filtered, params);
    std::copy(filtered.begin(),
              filtered.end(),
              reference.begin() + static_cast<std::ptrdiff_t>(offset));
  }
}

int main(int argc, char** argv) {
  const int width = argc > 1 ? std::atoi(argv[1]) : 1024;
  const int height = argc > 2 ? std::atoi(argv[2]) : width;
  const int patterns = argc > 3 ? std::atoi(argv[3]) : 4;
  const int threads_per_block = argc > 4 ? std::atoi(argv[4]) : 256;
  const int repeats = argc > 5 ? std::atoi(argv[5]) : 5;
  const int radius = 3;

  if (width <= 0 || height <= 0 || patterns <= 0 || threads_per_block <= 0 ||
      threads_per_block > 1024 || repeats <= 0 ||
      !valid_allocation_size(width, height, patterns)) {
    std::cerr << "error: invalid bilateral parameters\n";
    return 1;
  }
  if (!valid_cuda_grid_y(patterns)) {
    std::cerr << "error: invalid bilateral parameters\n";
    return 1;
  }

  BilateralParams params{width, height, radius, 9.0f, 900.0f};
  const size_t image_size = static_cast<size_t>(width) * static_cast<size_t>(height);
  const size_t count = image_size * static_cast<size_t>(patterns);
  const size_t bytes = count * sizeof(float);

  std::vector<float> input;
  std::vector<float> scalar_reference;
  std::vector<float> output(count, 0.0f);
  generate_patterns(input, width, height, patterns);
  scalar_patterns_reference(input, scalar_reference, params, patterns);

  float* d_input = nullptr;
  float* d_output = nullptr;
  cudaEvent_t kernel_start = nullptr;
  cudaEvent_t kernel_stop = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventCreate(&kernel_start));
  CUDA_CHECK(cudaEventCreate(&kernel_stop));

  const dim3 block(static_cast<unsigned int>(threads_per_block));
  const dim3 grid(static_cast<unsigned int>((image_size + threads_per_block - 1) / threads_per_block),
                  static_cast<unsigned int>(patterns));

  CUDA_CHECK(cudaEventRecord(kernel_start));
  for (int i = 0; i < repeats; ++i) {
    bilateral_multi_pattern_kernel<<<grid, block>>>(d_input, d_output, params, patterns);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaEventRecord(kernel_stop));
  CUDA_CHECK(cudaEventSynchronize(kernel_stop));

  float total_kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_kernel_ms, kernel_start, kernel_stop));
  const double avg_kernel_ms = static_cast<double>(total_kernel_ms) / static_cast<double>(repeats);
  const double avg_ms_per_pattern = avg_kernel_ms / static_cast<double>(patterns);

  CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
  const double checksum = checksum_image(output);
  const float diff = max_abs_diff(output, scalar_reference);

  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_stop));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));

  std::cout << "part=P5 multi_pattern_cuda\n";
  std::cout << "width=" << width << " height=" << height << " radius=" << radius
            << " patterns=" << patterns << " threads_per_block=" << threads_per_block
            << " repeats=" << repeats << "\n";
  std::cout << "checksum=" << std::fixed << std::setprecision(6) << checksum << "\n";
  std::cout << "max_abs_diff=" << std::fixed << std::setprecision(6) << diff << "\n";
  std::cout << "total_kernel_ms=" << std::fixed << std::setprecision(6) << total_kernel_ms
            << "\n";
  std::cout << "avg_kernel_ms=" << std::fixed << std::setprecision(6) << avg_kernel_ms << "\n";
  std::cout << "avg_ms_per_pattern=" << std::fixed << std::setprecision(6) << avg_ms_per_pattern
            << "\n";

  ensure_results_dir();
  std::ostringstream row;
  row << std::fixed << std::setprecision(6);
  row << "P5," << width << "," << height << "," << radius << "," << patterns << ","
      << threads_per_block << "," << repeats << "," << checksum << "," << diff << ","
      << avg_kernel_ms << "," << avg_ms_per_pattern << "," << total_kernel_ms;
  append_csv_line("results/p5_multi_pattern.csv", row.str());

  return diff <= 0.05f ? 0 : 2;
}
