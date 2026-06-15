#ifndef RESULT_IO_H
#define RESULT_IO_H

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

inline void ensure_results_dir() {
  std::filesystem::create_directories("results");
}

inline void append_csv_line(const std::string& path, const std::string& line) {
  std::ofstream out(path, std::ios::app);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open results file: " + path);
  }

  out << line << "\n";
  out.flush();
  if (!out) {
    throw std::runtime_error("failed to write results file: " + path);
  }
}

#endif
