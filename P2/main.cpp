#include "../common/bilateral_common.h"
#include "../common/result_io.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

static float reduce_sum_scalar(const float* values, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    sum += values[i];
  }
  return sum;
}

static float reduce_sum_rvv(const float* values, int n) {
#ifdef __riscv_vector
  float sum = 0.0f;
  int offset = 0;
  while (offset < n) {
    const size_t vl = __riscv_vsetvl_e32m1(static_cast<size_t>(n - offset));
    vfloat32m1_t chunk = __riscv_vle32_v_f32m1(values + offset, vl);
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t reduced = __riscv_vfredusum_vs_f32m1_f32m1(chunk, zero, vl);
    sum += __riscv_vfmv_f_s_f32m1_f32(reduced);
    offset += static_cast<int>(vl);
  }
  return sum;
#else
  return reduce_sum_scalar(values, n);
#endif
}

static float rvv_reduction_filter_one(const std::vector<float>& input,
                                      int x,
                                      int y,
                                      const BilateralParams& params) {
  const int width = params.width;
  const int height = params.height;
  const int radius = params.radius;
  const int window_width = 2 * radius + 1;
  const int window_size = window_width * window_width;
  const float center = input[static_cast<size_t>(y) * width + x];

  std::vector<float> numerator_terms(static_cast<size_t>(window_size), 0.0f);
  std::vector<float> denominator_terms(static_cast<size_t>(window_size), 0.0f);

  int term = 0;
  for (int dy = -radius; dy <= radius; ++dy) {
    const int yy = clamp_int(y + dy, 0, height - 1);
    for (int dx = -radius; dx <= radius; ++dx) {
      const int xx = clamp_int(x + dx, 0, width - 1);
      const float neighbor = input[static_cast<size_t>(yy) * width + xx];
      const float weight = bilateral_weight(dx, dy, center, neighbor, params);
      numerator_terms[static_cast<size_t>(term)] = neighbor * weight;
      denominator_terms[static_cast<size_t>(term)] = weight;
      ++term;
    }
  }

  const float numerator = reduce_sum_rvv(numerator_terms.data(), window_size);
  const float denominator = reduce_sum_rvv(denominator_terms.data(), window_size);
  return numerator / denominator;
}

static void rvv_reduction_filter(const std::vector<float>& input,
                                 std::vector<float>& output,
                                 const BilateralParams& params) {
  const size_t expected_size = validate_bilateral_input(input, params);
  output.assign(expected_size, 0.0f);
  for (int y = 0; y < params.height; ++y) {
    for (int x = 0; x < params.width; ++x) {
      output[static_cast<size_t>(y) * params.width + x] =
          rvv_reduction_filter_one(input, x, y, params);
    }
  }
}

int main(int argc, char** argv) {
  const int width = argc > 1 ? std::atoi(argv[1]) : 32;
  const int height = argc > 2 ? std::atoi(argv[2]) : width;
  const int radius = 3;
  if (width <= 0 || height <= 0) {
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
  rvv_reduction_filter(input, output, params);
  const auto stop = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(stop - start).count();

  const double checksum = checksum_image(output);
  const float diff = max_abs_diff(output, scalar_reference);
  std::cout << "part=P2 rvv_reduction\n";
  std::cout << "width=" << width << " height=" << height << " radius=" << radius << "\n";
  std::cout << "checksum=" << std::fixed << std::setprecision(6) << checksum << "\n";
  std::cout << "max_abs_diff=" << std::fixed << std::setprecision(6) << diff << "\n";
  std::cout << "host_fallback_ms=" << std::fixed << std::setprecision(6) << ms << "\n";
  print_selected_pixels("selected", output, width, height);

  ensure_results_dir();
  std::ostringstream row;
  row << std::fixed << std::setprecision(6);
  row << "P2," << width << "," << height << "," << radius << "," << checksum << "," << diff << ","
      << ms;
  append_csv_line("results/p2_rvv_reduction.csv", row.str());

  return diff <= 0.01f ? 0 : 2;
}
