#pragma once
#include "common.h"
namespace utils {

template <typename T>
bool search(const std::vector<T> &vlist, T key){
  return std::find(vlist.begin(), vlist.end(), key) != vlist.end();
}

inline void split(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ") {
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  std::string::size_type pos     = str.find_first_of(delimiters, lastPos);
  while (std::string::npos != pos || std::string::npos != lastPos) {
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    lastPos = str.find_first_not_of(delimiters, pos);
    pos = str.find_first_of(delimiters, lastPos);
  }
}

inline std::vector<int> PrefixSum(const std::vector<int> &degrees) {
  std::vector<int> sums(degrees.size() + 1);
  int total = 0;
  for (size_t n=0; n < degrees.size(); n++) {
    sums[n] = total;
    total += degrees[n];
  }
  sums[degrees.size()] = total;
  return sums;
}

}
