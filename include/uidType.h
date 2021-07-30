#pragma once

#include <cassert>
#include <limits>
#include <utility>
#include <string>
#include <vector>

// vertex ID type in a pattern graph
class uidType {
private:
    int64_t id;
public:
    static constexpr int64_t MIN = -1;
    static constexpr int64_t MAX = std::numeric_limits<int64_t>::max();

    void checkRep() {
        assert(id >= MIN && id <= MAX);
    }
    uidType(): id(0) {}
    uidType(int uid) : id(uid) {
        checkRep();
    }
    uidType(size_t uid) : id(uid) {
        checkRep();
    }
    uidType(int64_t uid) : id(uid) {
        checkRep();
    }

    int64_t value() const {
        return id;
    }

    bool operator==(const uidType &that) const {
        return id == that.id;
    }
    bool operator!=(const uidType &that) const {
        return id != that.id;
    }
    bool operator<(const uidType &that) const {
        return id < that.id;
    }
    bool operator >(const uidType &that) const {
        return id > that.id;
    }
    bool is_bounded() const {
        return id != MIN && id != MAX;
    }

    std::string str() const {
        if (id == MIN) {
            return "MIN";
        }
        if (id == MAX) {
            return "MAX";
        }
        return "u" + std::to_string(id);
    }

    static const std::pair<uidType, uidType>
    unbounded() {
        return std::make_pair(MIN, MAX);
    }

    static const std::vector<uidType>
    range(uidType lower, uidType upper) {
        if (upper < lower || lower == upper)
            return {};

        std::vector<uidType> range;
        for (auto id = lower.value(); id < upper.value(); ++id) {
            range.push_back(id);
        }
        return range;
    }
};

namespace std {
    template <>
    struct less<uidType> {
        bool operator() (const uidType &lhs, const uidType &rhs) const {
            return lhs.value() < rhs.value();
        }
    };
}

