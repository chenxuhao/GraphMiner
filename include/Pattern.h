#pragma once

#include <utility>
#include <vector>
#include <unordered_map>
#include "uidType.h"
#include "ExecutionPlan.h"

namespace EdgeInduced {
    using namespace std;

    // edge centric embeddings
    static vector<Rule> wedge = {
        {uidType::unbounded(), {0}, {}},
        {make_pair(uidType::MIN, 1), {0}, {}}
    };

    static vector<Rule> triangle = {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {make_pair(uidType::MIN, 1), {0, 1}, {}}
    };

    static vector<Rule> diamond = {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {uidType::unbounded(), {0, 1}, {}},
        {make_pair(uidType::MIN, 2), {0, 1}, {}}
    };

    static vector<Rule> three_star = {
        {uidType::unbounded(), {0}, {}},
        {make_pair(uidType::MIN, 1), {0}, {}},
        {make_pair(uidType::MIN, 2), {0}, {}}
    };

    static vector<Rule> four_cycle = {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {make_pair(uidType::MIN, 1), {0}, {}},
        {make_pair(uidType::MIN, 0), {1, 2}, {}}
    };

    static vector<Rule> four_path = {
        {uidType::unbounded(), {0}, {}},
        {make_pair(uidType::MIN, 0), {0}, {}},
        {uidType::unbounded(), {2}, {}}
    };

    static vector<Rule> tail_triangle = {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {uidType::unbounded(), {0, 1}, {}},
        {uidType::unbounded(), {2}, {}}
    };

    static vector<Rule> four_clique = {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {make_pair(uidType::MIN, 1), {0, 1}, {}},
        {make_pair(uidType::MIN, 2), {0, 1, 2}, {}}
    };

    static vector<Rule> house = {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {uidType::unbounded(), {0, 1}, {}},
        {uidType::unbounded(), {1}, {}},
        {uidType::unbounded(), {0, 3}, {}}
    };

    static vector<Rule> five_clique {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {make_pair(uidType::MIN, 1), {0, 1}, {}},
        {make_pair(uidType::MIN, 2), {0, 1, 2}, {}},
        {make_pair(uidType::MIN, 3), {0, 1, 2, 3}, {}},
    };

    static vector<Rule> _6_clique {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {make_pair(uidType::MIN, 1), {0, 1}, {}},
        {make_pair(uidType::MIN, 2), {0, 1, 2}, {}},
        {make_pair(uidType::MIN, 3), {0, 1, 2, 3}, {}},
        {make_pair(uidType::MIN, 4), {0, 1, 2, 3, 4}, {}}
    };

    static vector<Rule> _7_clique {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {make_pair(uidType::MIN, 1), {0, 1}, {}},
        {make_pair(uidType::MIN, 2), {0, 1, 2}, {}},
        {make_pair(uidType::MIN, 3), {0, 1, 2, 3}, {}},
        {make_pair(uidType::MIN, 4), {0, 1, 2, 3, 4}, {}},
        {make_pair(uidType::MIN, 5), {0, 1, 2, 3, 4, 5}, {}}
    };

    static vector<Rule> _8_clique {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {make_pair(uidType::MIN, 1), {0, 1}, {}},
        {make_pair(uidType::MIN, 2), {0, 1, 2}, {}},
        {make_pair(uidType::MIN, 3), {0, 1, 2, 3}, {}},
        {make_pair(uidType::MIN, 4), {0, 1, 2, 3, 4}, {}},
        {make_pair(uidType::MIN, 5), {0, 1, 2, 3, 4, 5}, {}},
        {make_pair(uidType::MIN, 6), {0, 1, 2, 3, 4, 5, 6}, {}}
    };

    static vector<Rule> _9_clique {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {make_pair(uidType::MIN, 1), {0, 1}, {}},
        {make_pair(uidType::MIN, 2), {0, 1, 2}, {}},
        {make_pair(uidType::MIN, 3), {0, 1, 2, 3}, {}},
        {make_pair(uidType::MIN, 4), {0, 1, 2, 3, 4}, {}},
        {make_pair(uidType::MIN, 5), {0, 1, 2, 3, 4, 5}, {}},
        {make_pair(uidType::MIN, 6), {0, 1, 2, 3, 4, 5, 6}, {}},
        {make_pair(uidType::MIN, 7), {0, 1, 2, 3, 4, 5, 6, 7}, {}}
    };

    static const unordered_map<string, vector<Rule>> pattern_map {
      {"3-clique", triangle},
      {"4-clique", four_clique},
      {"5-clique", five_clique},
      {"6-clique", _6_clique},
      {"7-clique", _7_clique},
      {"8-clique", _8_clique},
      {"9-clique", _9_clique},
      {"wedge", wedge},
      {"triangle", triangle},
      {"diamond", diamond},
      {"3-star", three_star},
      {"4-cycle", four_cycle},
      {"4-path", four_path},
      {"tailed-triangle", tail_triangle},
      {"house", house}
    };

    static const vector<Rule> rules_of_pattern(const string pattern_name) {
      return pattern_map.at(pattern_name);
    }
};

namespace VertexInduced {
    using namespace std;

    static vector<Rule> wedge = {
        {uidType::unbounded(), {0}, {}},
        {make_pair(uidType::MIN, 1), {0}, {1}}
    };

    static vector<Rule> triangle = {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {make_pair(uidType::MIN, 1), {0, 1}, {}}
    };

    static vector<Rule> diamond = {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {uidType::unbounded(), {0, 1}, {}},
        {make_pair(uidType::MIN, 2), {0, 1}, {2}}
    };

    static vector<Rule> three_star = {
        {uidType::unbounded(), {0}, {}},
        {make_pair(uidType::MIN, 1), {0}, {1}},
        {make_pair(uidType::MIN, 2), {0}, {1, 2}}
    };

    static vector<Rule> four_cycle = {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {make_pair(uidType::MIN, 1), {0}, {1}},
        {make_pair(uidType::MIN, 0), {1, 2}, {0}}
    };

    static vector<Rule> four_path = {
        {uidType::unbounded(), {0}, {}},
        {make_pair(uidType::MIN, 0), {0}, {1}},
        {uidType::unbounded(), {2}, {0, 1}}
    };

    static vector<Rule> tail_triangle = {
        // {uidType::unbounded(), {0}, {}},
        // {uidType::unbounded(), {0}, {1}},
        // {make_pair(uidType::MIN, 2), {0, 2}, {1}}
        {make_pair(uidType::MIN, 0), {0}, {}},
        {uidType::unbounded(), {0, 1}, {}},
        {uidType::unbounded(), {2}, {0, 1}}
    };

    static vector<Rule> four_clique = {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {make_pair(uidType::MIN, 1), {0, 1}, {}},
        {make_pair(uidType::MIN, 2), {0, 1, 2}, {}}
    };

    static vector<Rule> house = {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {uidType::unbounded(), {0, 1}, {}},
        {uidType::unbounded(), {1}, {2, 0}},
        {uidType::unbounded(), {0, 3}, {2, 1}}
    };

    static vector<Rule> five_clique {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {make_pair(uidType::MIN, 1), {0, 1}, {}},
        {make_pair(uidType::MIN, 2), {0, 1, 2}, {}},
        {make_pair(uidType::MIN, 3), {0, 1, 2, 3}, {}},
    };

    static vector<Rule> _6_clique {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {make_pair(uidType::MIN, 1), {0, 1}, {}},
        {make_pair(uidType::MIN, 2), {0, 1, 2}, {}},
        {make_pair(uidType::MIN, 3), {0, 1, 2, 3}, {}},
        {make_pair(uidType::MIN, 4), {0, 1, 2, 3, 4}, {}}
    };

    static vector<Rule> _7_clique {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {make_pair(uidType::MIN, 1), {0, 1}, {}},
        {make_pair(uidType::MIN, 2), {0, 1, 2}, {}},
        {make_pair(uidType::MIN, 3), {0, 1, 2, 3}, {}},
        {make_pair(uidType::MIN, 4), {0, 1, 2, 3, 4}, {}},
        {make_pair(uidType::MIN, 5), {0, 1, 2, 3, 4, 5}, {}}
    };

    static vector<Rule> _8_clique {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {make_pair(uidType::MIN, 1), {0, 1}, {}},
        {make_pair(uidType::MIN, 2), {0, 1, 2}, {}},
        {make_pair(uidType::MIN, 3), {0, 1, 2, 3}, {}},
        {make_pair(uidType::MIN, 4), {0, 1, 2, 3, 4}, {}},
        {make_pair(uidType::MIN, 5), {0, 1, 2, 3, 4, 5}, {}},
        {make_pair(uidType::MIN, 6), {0, 1, 2, 3, 4, 5, 6}, {}}
    };

    static vector<Rule> _9_clique {
        {make_pair(uidType::MIN, 0), {0}, {}},
        {make_pair(uidType::MIN, 1), {0, 1}, {}},
        {make_pair(uidType::MIN, 2), {0, 1, 2}, {}},
        {make_pair(uidType::MIN, 3), {0, 1, 2, 3}, {}},
        {make_pair(uidType::MIN, 4), {0, 1, 2, 3, 4}, {}},
        {make_pair(uidType::MIN, 5), {0, 1, 2, 3, 4, 5}, {}},
        {make_pair(uidType::MIN, 6), {0, 1, 2, 3, 4, 5, 6}, {}},
        {make_pair(uidType::MIN, 7), {0, 1, 2, 3, 4, 5, 6, 7}, {}}
    };

    static const unordered_map<string, vector<Rule>> pattern_map {
      {"3-clique", triangle},
      {"4-clique", four_clique},
      {"5-clique", five_clique},
      {"6-clique", _6_clique},
      {"7-clique", _7_clique},
      {"8-clique", _8_clique},
      {"9-clique", _9_clique},
      {"wedge", wedge},
      {"triangle", triangle},
      {"diamond", diamond},
      {"3-star", three_star},
      {"4-cycle", four_cycle},
      {"4-path", four_path},
      {"tailed-triangle", tail_triangle},
      {"house", house}
    };

    static const vector<Rule> rules_of_pattern(const string pattern_name) {
      return pattern_map.at(pattern_name);
    }

};
