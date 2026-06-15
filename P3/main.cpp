#include "../common/bilateral_common.h"
#include "../common/result_io.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

static void simd_like_filter_scalar_fallback(const std::vector<float>& input,
                                             std::vector<float>& output,
                                             const BilateralParams& params,
                                             int k) {
  validate_bilateral_input(input, params);
  output.assign(input.size(), 0.0f);

  for (int x = 0; x < params.width; ++x) {
    for (int y = 0; y < params.height; y += k) {
      const int lanes = std::min(k, params.height - y);
      for (int lane = 0; lane < lanes; ++lane) {
        const int yy = y + lane;
        output[static_cast<size_t>(yy) * params.width + x] =
            scalar_filter_one(input, x, yy, params);
      }
    }
  }
}

static void simd_like_filter_rvv(const std::vector<float>& input,
                                 std::vector<float>& output,
                                 const BilateralParams& params,
                                 int k) {
#ifdef __riscv_vector
  scalar_bilateral_filter(input, output, params);

  const int width = params.width;
  const int height = params.height;
  const int radius = params.radius;
  const std::ptrdiff_t row_stride_bytes = static_cast<std::ptrdiff_t>(width * sizeof(float));

  if (width <= 2 * radius || height <= 2 * radius) {
    return;
  }

  for (int x = radius; x < width - radius; ++x) {
    int y = radius;
    while (y < height - radius) {
      const int lane_limit = std::min(k, height - radius - y);
      const size_t vl = __riscv_vsetvl_e32m1(static_cast<size_t>(lane_limit));

      const float* center_base = input.data() + static_cast<size_t>(y) * width + x;
      vfloat32m1_t center = __riscv_vlse32_v_f32m1(center_base, row_stride_bytes, vl);
      vfloat32m1_t numerator = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      vfloat32m1_t denominator = __riscv_vfmv_v_f_f32m1(0.0f, vl);

      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
          const float* neighbor_base =
              input.data() + static_cast<size_t>(y + dy) * width + (x + dx);
          vfloat32m1_t neighbor = __riscv_vlse32_v_f32m1(neighbor_base, row_stride_bytes, vl);
          vfloat32m1_t diff = __riscv_vfsub_vv_f32m1(center, neighbor, vl);
          vfloat32m1_t diff2 = __riscv_vfmul_vv_f32m1(diff, diff, vl);
          vfloat32m1_t range_scaled = __riscv_vfdiv_vf_f32m1(diff2, params.sigma_r2, vl);
          vfloat32m1_t range_den = __riscv_vfadd_vf_f32m1(range_scaled, 1.0f, vl);
          vfloat32m1_t range = __riscv_vfrdiv_vf_f32m1(range_den, 1.0f, vl);
          const float spatial = spatial_weight(dx, dy, params.sigma_s2);
          vfloat32m1_t weight = __riscv_vfmul_vf_f32m1(range, spatial, vl);

          numerator = __riscv_vfmacc_vv_f32m1(numerator, neighbor, weight, vl);
          denominator = __riscv_vfadd_vv_f32m1(denominator, weight, vl);
        }
      }

      vfloat32m1_t result = __riscv_vfdiv_vv_f32m1(numerator, denominator, vl);
      float* out_base = output.data() + static_cast<size_t>(y) * width + x;
      __riscv_vsse32_v_f32m1(out_base, row_stride_bytes, result, vl);

      y += static_cast<int>(vl);
    }
  }
#else
  simd_like_filter_scalar_fallback(input, output, params, k);
#endif
}

int main(int argc, char** argv) {
  const int width = argc > 1 ? std::atoi(argv[1]) : 32;
  const int height = argc > 2 ? std::atoi(argv[2]) : width;
  const int k = argc > 3 ? std::atoi(argv[3]) : 4;
  const int radius = 3;
  if (width <= 0 || height <= 0 || k <= 0) {
    std::cerr << "error: invalid bilateral parameters\n";
    return 1;
  }

  BilateralParams params{width, height, radius, 9.0f, 900.0f};

  std::vector<float> input;
  std::vector<float> scalar_reference;
  std::vector<float> output;
  generate_image(input, width, height, 0);
  scalar_bilateral_filter(input, scalar_reference, params);

  const auto start = std::chrono::high_resolution_clock::now();
  simd_like_filter_rvv(input, output, params, k);
  const auto stop = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(stop - start).count();

  const double checksum = checksum_image(output);
  const float diff = max_abs_diff(output, scalar_reference);
  std::cout << "part=P3 simd_like_rvv\n";
  std::cout << "width=" << width << " height=" << height << " radius=" << radius << " k=" << k
            << "\n";
  std::cout << "checksum=" << std::fixed << std::setprecision(6) << checksum << "\n";
  std::cout << "max_abs_diff=" << std::fixed << std::setprecision(6) << diff << "\n";
  std::cout << "host_fallback_ms=" << std::fixed << std::setprecision(6) << ms << "\n";
  print_selected_pixels("selected", output, width, height);

  ensure_results_dir();
  std::ostringstream row;
  row << std::fixed << std::setprecision(6);
  row << "P3," << width << "," << height << "," << radius << "," << k << "," << checksum << ","
      << diff << "," << ms;
  append_csv_line("results/p3_simd_like_rvv.csv", row.str());

  return diff <= 0.01f ? 0 : 2;
}
