#pragma once

#include <sstream>
#include <string>
#include <vector>

// Add this helper function at the top of the file, after the includes
template <typename T>
std::string vector_to_string(const std::vector<T>& vec) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    oss << vec[i];
    if (i < vec.size() - 1)
      oss << ", ";
  }
  oss << "]";
  return oss.str();
}
