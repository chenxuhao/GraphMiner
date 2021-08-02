#pragma once

#include <string>

class Pattern {
public:
  Pattern() {}
  Pattern(std::string name) : name_(name) { }
  ~Pattern() {}
  bool is_diamond() { return name_ == "diamond"; }
  bool is_rectangle() { return name_ == "rectangle"; }
  bool is_pentagon() { return name_ == "pentagon"; }
  bool is_house() { return name_ == "house"; }
  std::string to_string() { return name_; }
  std::string get_name() { return name_; }
private:
  std::string name_;
};

