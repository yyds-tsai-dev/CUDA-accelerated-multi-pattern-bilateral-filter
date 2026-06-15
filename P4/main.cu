#include "../common/bilateral_common.h"
#include "../common/result_io.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
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

__global__ void bilateral_naive_kernel(const float* input, float* output, BilateralParams params) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= params.width || y >= params.height) {
    return;
  }

  const float center = input[static_cast<size_t>(y) * params.width + x];
  float numerator = 0.0f;
  float denominator = 0.0f;

  for (int dy = -params.radius; dy <= params.radius; ++dy) {
    const int yy = d_clamp_int(y + dy, 0, params.height - 1);
    for (int dx = -params.radius; dx <= params.radius; ++dx) {
      const int xx = d_clamp_int(x + dx, 0, params.width - 1);
      const float neighbor = input[static_cast<size_t>(yy) * params.width + xx];
      const float weight = d_spatial_weight(dx, dy, params.sigma_s2) *
                           d_range_weight(center, neighbor, params.sigma_r2);
      numerator += neighbor * weight;
      denominator += weight;
    }
  }

  output[static_cast<size_t>(y) * params.width + x] = numerator / denominator;
}

__global__ void bilateral_shared_kernel(const float* input, float* output, BilateralParams params) {
  extern __shared__ float tile[];

  const int radius = params.radius;
  const int tile_width = blockDim.x + 2 * radius;
  const int tile_height = blockDim.y + 2 * radius;
  const int tile_origin_x = blockIdx.x * blockDim.x - radius;
  const int tile_origin_y = blockIdx.y * blockDim.y - radius;

  for (int tile_y = threadIdx.y; tile_y < tile_height; tile_y += blockDim.y) {
    const int global_y = d_clamp_int(tile_origin_y + tile_y, 0, params.height - 1);
    for (int tile_x = threadIdx.x; tile_x < tile_width; tile_x += blockDim.x) {
      const int global_x = d_clamp_int(tile_origin_x + tile_x, 0, params.width - 1);
      tile[tile_y * tile_width + tile_x] =
          input[static_cast<size_t>(global_y) * params.width + global_x];
    }
  }

  __syncthreads();

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= params.width || y >= params.height) {
    return;
  }

  const int local_x = threadIdx.x + radius;
  const int local_y = threadIdx.y + radius;
  const float center = tile[local_y * tile_width + local_x];
  float numerator = 0.0f;
  float denominator = 0.0f;

  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      const float neighbor = tile[(local_y + dy) * tile_width + (local_x + dx)];
      const float weight =
          d_spatial_weight(dx, dy, params.sigma_s2) * d_range_weight(center, neighbor, params.sigma_r2);
      numerator += neighbor * weight;
      denominator += weight;
    }
  }

  output[static_cast<size_t>(y) * params.width + x] = numerator / denominator;
}

static double run_kernel(bool shared,
                         const std::vector<float>& input,
                         std::vector<float>& output,
                         const BilateralParams& params,
                         dim3 block,
                         int repeats) {
  const size_t count = input.size();
  const size_t bytes = count * sizeof(float);
  output.assign(count, 0.0f);

  float* d_input = nullptr;
  float* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));

  cudaEvent_t h2d_start = nullptr;
  cudaEvent_t h2d_stop = nullptr;
  cudaEvent_t kernel_start = nullptr;
  cudaEvent_t kernel_stop = nullptr;
  cudaEvent_t d2h_start = nullptr;
  cudaEvent_t d2h_stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&h2d_start));
  CUDA_CHECK(cudaEventCreate(&h2d_stop));
  CUDA_CHECK(cudaEventCreate(&kernel_start));
  CUDA_CHECK(cudaEventCreate(&kernel_stop));
  CUDA_CHECK(cudaEventCreate(&d2h_start));
  CUDA_CHECK(cudaEventCreate(&d2h_stop));

  CUDA_CHECK(cudaEventRecord(h2d_start));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(h2d_stop));
  CUDA_CHECK(cudaEventSynchronize(h2d_stop));
  float h2d_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop));

  const dim3 grid((params.width + block.x - 1) / block.x, (params.height + block.y - 1) / block.y);
  const size_t shared_bytes =
      static_cast<size_t>(block.x + 2 * params.radius) * (block.y + 2 * params.radius) * sizeof(float);

  CUDA_CHECK(cudaEventRecord(kernel_start));
  for (int i = 0; i < repeats; ++i) {
    if (shared) {
      bilateral_shared_kernel<<<grid, block, shared_bytes>>>(d_input, d_output, params);
    } else {
      bilateral_naive_kernel<<<grid, block>>>(d_input, d_output, params);
    }
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaEventRecord(kernel_stop));
  CUDA_CHECK(cudaEventSynchronize(kernel_stop));
  float total_kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_kernel_ms, kernel_start, kernel_stop));
  const double avg_kernel_ms = static_cast<double>(total_kernel_ms) / static_cast<double>(repeats);

  CUDA_CHECK(cudaEventRecord(d2h_start));
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(d2h_stop));
  CUDA_CHECK(cudaEventSynchronize(d2h_stop));
  float d2h_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop));

  std::cout << "h2d_ms=" << std::fixed << std::setprecision(6) << h2d_ms
            << " avg_kernel_ms=" << avg_kernel_ms << " d2h_ms=" << d2h_ms
            << " repeats=" << repeats << "\n";

  CUDA_CHECK(cudaEventDestroy(h2d_start));
  CUDA_CHECK(cudaEventDestroy(h2d_stop));
  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_stop));
  CUDA_CHECK(cudaEventDestroy(d2h_start));
  CUDA_CHECK(cudaEventDestroy(d2h_stop));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));

  return avg_kernel_ms;
}

int main(int argc, char** argv) {
  const int width = argc > 1 ? std::atoi(argv[1]) : 512;
  const int height = argc > 2 ? std::atoi(argv[2]) : width;
  const int radius = argc > 3 ? std::atoi(argv[3]) : 3;
  const int block_x = argc > 4 ? std::atoi(argv[4]) : 16;
  const int block_y = argc > 5 ? std::atoi(argv[5]) : 16;
  const bool shared = argc > 6 ? std::atoi(argv[6]) != 0 : false;
  const int repeats = argc > 7 ? std::atoi(argv[7]) : 5;

  if (width <= 0 || height <= 0 || radius < 0 || block_x <= 0 || block_y <= 0 || repeats <= 0) {
    std::cerr << "error: invalid bilateral parameters\n";
    return 1;
  }

  BilateralParams params{width, height, radius, 9.0f, 900.0f};

  std::vector<float> input;
  std::vector<float> scalar_reference;
  std::vector<float> output;
  generate_image(input, width, height, 0);
  scalar_bilateral_filter(input, scalar_reference, params);

  const double avg_kernel_ms =
      run_kernel(shared, input, output, params, dim3(block_x, block_y), repeats);

  const double checksum = checksum_image(output);
  const float diff = max_abs_diff(output, scalar_reference);
  const char* variant = shared ? "shared" : "naive";

  std::cout << "part=P4 cuda_simt variant=" << variant << "\n";
  std::cout << "width=" << width << " height=" << height << " radius=" << radius << " block="
            << block_x << "x" << block_y << " repeats=" << repeats << "\n";
  std::cout << "checksum=" << std::fixed << std::setprecision(6) << checksum << "\n";
  std::cout << "max_abs_diff=" << std::fixed << std::setprecision(6) << diff << "\n";
  print_selected_pixels("selected", output, width, height);

  ensure_results_dir();
  std::ostringstream row;
  row << std::fixed << std::setprecision(6);
  row << "P4," << variant << "," << width << "," << height << "," << radius << "," << block_x
      << "x" << block_y << "," << repeats << "," << checksum << "," << diff << ","
      << avg_kernel_ms;
  append_csv_line("results/p4_cuda.csv", row.str());

  return diff <= 0.05f ? 0 : 2;
}
