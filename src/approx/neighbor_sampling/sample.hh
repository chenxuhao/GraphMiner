#pragma once 

#include <locale>
#include <random>
#include <iomanip>

template<class T>
std::string FormatWithCommas(T value) {
  std::stringstream ss;
  ss.imbue(std::locale(""));
  ss << std::fixed << value;
  return ss.str();
}

template <typename T>
T random_select_single(T begin, T end) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<T> dist(begin, end);
  return dist(gen);
}

template <typename T>
void random_select_batch(T begin, T end, int64_t n, std::vector<T> &samples) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<T> dist(begin, end);
  for (int64_t i = 0; i < n; i++)
    samples[i] = dist(gen);
}

