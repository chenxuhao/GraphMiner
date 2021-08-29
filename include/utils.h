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

}
