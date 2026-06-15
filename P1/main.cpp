#include "../common/bilateral_common.h"
#include "../common/result_io.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

int main(int argc, char** argv) {
  const int width = argc > 1 ? std::atoi(argv[1]) : 32;
  const int height = argc > 2 ? std::atoi(argv[2]) : width;
  const int radius = argc > 3 ? std::atoi(argv[3]) : 3;
  BilateralParams params{width, height, radius, 9.0f, 900.0f};

  std::vector<float> input;
  std::vector<float> output;
  generate_image(input, width, height, 0);

  const auto start = std::chrono::high_resolution_clock::now();
  scalar_bilateral_filter(input, output, params);
  const auto stop = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(stop - start).count();

  const double checksum = checksum_image(output);
  std::cout << "part=P1 scalar\n";
  std::cout << "width=" << width << " height=" << height << " radius=" << radius << "\n";
  std::cout << "checksum=" << std::fixed << std::setprecision(6) << checksum << "\n";
  std::cout << "host_ms=" << std::fixed << std::setprecision(6) << ms << "\n";
  print_selected_pixels("selected", output, width, height);

  ensure_results_dir();
  std::ostringstream row;
  row << "P1," << width << "," << height << "," << radius << "," << checksum << "," << ms;
  append_csv_line("results/p1_scalar.csv", row.str());
  write_pgm("data/p1_scalar_output.pgm", output, width, height);
  return 0;
}
