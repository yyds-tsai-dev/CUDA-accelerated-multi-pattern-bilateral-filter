#ifndef BILATERAL_COMMON_H
#define BILATERAL_COMMON_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct BilateralParams {
  int width;
  int height;
  int radius;
  float sigma_s2;
  float sigma_r2;
};

inline int clamp_int(int v, int lo, int hi) {
  return std::max(lo, std::min(v, hi));
}

inline float clean_pixel_value(int x, int y, int pattern) {
  const int base = (x * 13 + y * 17 + pattern * 19) & 255;
  const int edge = (x > 0 && ((x / 16) & 1)) ? 42 : 0;
  return static_cast<float>((base + edge) & 255);
}

inline float noisy_pixel_value(int x, int y, int pattern) {
  const int noise = ((x * 31 + y * 7 + pattern * 23) % 21) - 10;
  const int value = static_cast<int>(clean_pixel_value(x, y, pattern)) + noise;
  return static_cast<float>(clamp_int(value, 0, 255));
}

inline void generate_image(std::vector<float>& image, int width, int height, int pattern) {
  image.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0.0f);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      image[static_cast<size_t>(y) * width + x] = noisy_pixel_value(x, y, pattern);
    }
  }
}

inline float spatial_weight(int dx, int dy, float sigma_s2) {
  const float d2 = static_cast<float>(dx * dx + dy * dy);
  return 1.0f / (1.0f + d2 / sigma_s2);
}

inline float range_weight(float center, float neighbor, float sigma_r2) {
  const float diff = center - neighbor;
  return 1.0f / (1.0f + (diff * diff) / sigma_r2);
}

inline float bilateral_weight(int dx, int dy, float center, float neighbor, const BilateralParams& params) {
  return spatial_weight(dx, dy, params.sigma_s2) * range_weight(center, neighbor, params.sigma_r2);
}

inline size_t validate_bilateral_input(const std::vector<float>& input, const BilateralParams& params) {
  if (params.width <= 0 || params.height <= 0 || params.radius < 0) {
    throw std::invalid_argument("invalid bilateral parameters");
  }

  const size_t expected_size = static_cast<size_t>(params.width) * static_cast<size_t>(params.height);
  if (input.size() != expected_size) {
    throw std::invalid_argument("input size does not match bilateral parameters");
  }

  return expected_size;
}

inline float scalar_filter_one(const std::vector<float>& input, int x, int y, const BilateralParams& params) {
  validate_bilateral_input(input, params);
  if (x < 0 || x >= params.width || y < 0 || y >= params.height) {
    throw std::invalid_argument("filter coordinates out of range");
  }

  const int width = params.width;
  const int height = params.height;
  const float center = input[static_cast<size_t>(y) * width + x];
  float numerator = 0.0f;
  float denominator = 0.0f;

  for (int dy = -params.radius; dy <= params.radius; ++dy) {
    const int yy = clamp_int(y + dy, 0, height - 1);
    for (int dx = -params.radius; dx <= params.radius; ++dx) {
      const int xx = clamp_int(x + dx, 0, width - 1);
      const float neighbor = input[static_cast<size_t>(yy) * width + xx];
      const float weight = bilateral_weight(dx, dy, center, neighbor, params);
      numerator += neighbor * weight;
      denominator += weight;
    }
  }

  return numerator / denominator;
}

inline void scalar_bilateral_filter(const std::vector<float>& input,
                                    std::vector<float>& output,
                                    const BilateralParams& params) {
  const size_t expected_size = validate_bilateral_input(input, params);
  output.assign(expected_size, 0.0f);
  for (int y = 0; y < params.height; ++y) {
    for (int x = 0; x < params.width; ++x) {
      output[static_cast<size_t>(y) * params.width + x] = scalar_filter_one(input, x, y, params);
    }
  }
}

inline double checksum_image(const std::vector<float>& image) {
  double checksum = 0.0;
  for (size_t i = 0; i < image.size(); ++i) {
    checksum += static_cast<double>(image[i]) * static_cast<double>((i % 97) + 1);
  }
  return checksum;
}

inline float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("image sizes do not match");
  }

  float max_diff = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
  }
  return max_diff;
}

inline void print_selected_pixels(const char* label, const std::vector<float>& image, int width, int height) {
  const int xs[4] = {0, width / 3, (2 * width) / 3, width - 1};
  const int ys[4] = {0, height / 3, (2 * height) / 3, height - 1};
  std::cout << label;
  for (int i = 0; i < 4; ++i) {
    const int x = clamp_int(xs[i], 0, width - 1);
    const int y = clamp_int(ys[i], 0, height - 1);
    std::cout << " p" << i << "=" << std::fixed << std::setprecision(4)
              << image[static_cast<size_t>(y) * width + x];
  }
  std::cout << "\n";
}

inline void write_pgm(const std::string& path, const std::vector<float>& image, int width, int height) {
  std::ofstream out(path, std::ios::binary);
  out << "P5\n" << width << " " << height << "\n255\n";
  for (float value : image) {
    const int rounded = clamp_int(static_cast<int>(std::lround(value)), 0, 255);
    const unsigned char byte = static_cast<unsigned char>(rounded);
    out.write(reinterpret_cast<const char*>(&byte), 1);
  }
}

#endif
