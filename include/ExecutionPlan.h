#pragma once

#include <cassert>
#include <utility>
#include <optional>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <bitset>
#include <algorithm>

#include "VertexSet.h"
#include "cmap.h"
#include "uidType.h"

// A rule for extending a vertex
struct Rule {
    // partial order
    std::pair<uidType, uidType> bounds;
    // connection info
    std::set<uidType> connected;
    std::set<uidType> disconnected;

    std::string str() const {
        std::stringstream ss;

        ss << "Rule([" << bounds.first.str() << ", " << bounds.second.str() << "), ";

        ss << "{";
        std::string sep = "";
        for (auto uid : connected) {
            ss << sep << uid.str();
            sep = ",";
        }
        ss << "}, ";

        ss << "{";
        sep = "";
        for (auto uid : disconnected) {
            ss << sep << uid.str();
            sep = ",";
        }
        ss << "})";
        return ss.str();
    }
};

struct RuleReuse {
    std::optional<uidType> reuse;
    Rule rule;
    RuleReuse() {}
    RuleReuse(const Rule &rule): rule(rule) {}
    std::string str() const {
        std::stringstream ss;

        ss << "Rule(" << (reuse.has_value() ? reuse->str() : "<null>")
          << " [" << rule.bounds.first.str() << ", " << rule.bounds.second.str() << "), ";
        ss << "{";
        std::string sep = "";
        for (auto uid : rule.connected) {
            ss << sep << uid.str();
            sep = ",";
        }
        ss << "}, ";

        ss << "{";
        sep = "";
        for (auto uid : rule.disconnected) {
            ss << sep << uid.str();
            sep = ",";
        }
        ss << "})";
        return ss.str();
    }
};

using RulesReuse = std::vector<RuleReuse>;

class Rules {
private:
    std::vector<Rule> rules;

public:
    Rules() = default;
    Rules(const std::vector<Rule> &rules): rules(rules)
    {}

    Rule &operator[](size_t idx) { return rules[idx]; }
    const Rule &operator[](size_t idx) const { return rules[idx]; }

    size_t size() const { return rules.size(); }

    const Rule &of(uidType uid) const {
        return rules.at(uid.value()-1);
    }

    // build support map: uid -> uids that requires connection info of uid
    std::map<uidType, std::set<uidType>> get_support_map() const {

        std::map<uidType, std::set<uidType>> support_map;

        for (size_t i = 0; i < rules.size(); ++i) {
            support_map.insert({i, {}});
            for (auto uid : rules[i].connected) {
                support_map[uid].insert(i + 1);
            }
            for (auto uid : rules[i].disconnected) {
                support_map[uid].insert(i + 1);
            }
        }

        support_map.insert({rules.size(), {}});

        return support_map;
    }

    // ui <= uj ?
    // @param   ui, uj  two uids (MIN, MAX, or from rules)
    // @return  true if ui are always less or equal than uj in the pattern;
    //          false otherwise
    bool uid_le(uidType ui, uidType uj) const {
        // special cases
        if (ui == uj || ui == uidType::MIN || uj == uidType::MAX) {
            return true;
        } else if (ui == uidType::MAX || uj == uidType::MIN) {
            return false;
        }

        // ui != uj, and neither is MIN or MAX
        if (ui == 0) {
            return uid_le(ui, this->of(uj).bounds.first);
        } else if (uj == 0) {
            return uid_le(this->of(ui).bounds.second, uj);
        } else {
            return uid_le(ui, this->of(uj).bounds.first) || uid_le(this->of(ui).bounds.second, uj);
        }
    }

    // ui >= uj ?
    // @param   ui, uj  two uids (MIN, MAX, or from rules)
    // @return  true if ui are always greater or equal than uj in the pattern;
    //          false otherwise
    bool uid_ge(uidType ui, uidType uj) const {
        // special cases
        if (ui == uj || ui == uidType::MAX || uj == uidType::MIN) {
            return true;
        } else if (ui == uidType::MIN || uj == uidType::MAX) {
            return false;
        }

        // ui != uj, and neither is MIN or MAX
        if (ui == 0) {
            return uid_ge(ui, this->of(uj).bounds.second);
        } else if (uj == 0) {
            return uid_ge(this->of(ui).bounds.first, uj);
        } else {
            return uid_ge(ui, this->of(uj).bounds.second) || uid_ge(this->of(ui).bounds.first, uj);
        }
    }

    // @param   ui  the uid
    // @return  true if the rule this->of(ui) is simple, i.e. just an edgelist of
    //          a prev vertex with some bounds, but no other set intersection
    //          or difference involved
    bool is_rule_simple(uidType ui) const {
      const auto &ri = this->of(ui);
      return ri.connected.size() == 1 && ri.disconnected.size() == 0;
    }

    // @param   ui, uj      two uids from rules
    // @return  whether vertices represented by this->of(ui) is a subset of
    //          vertices represented by this->of(uj)
    bool is_subset(uidType ui, uidType uj) const {
        const auto &ri = this->of(ui);
        const auto &rj = this->of(uj);

        if (!std::includes(ri.connected.begin(), ri.connected.end(),
                           rj.connected.begin(), rj.connected.end()))
            return false;
        if (!std::includes(ri.disconnected.begin(), ri.disconnected.end(),
                           rj.disconnected.begin(), rj.disconnected.end()))
            return false;

        return uid_le(rj.bounds.first, ri.bounds.first) && uid_le(ri.bounds.second, rj.bounds.second);
    }

    // @param   dependents  a set of uids from rules
    // @param   uid
    // @return  a "superset" rule so that: for each uid in dependents,
    //          the vertices represented by this->of(uid) are a subset of
    //          vertices represented by the returned "superset" rule
    Rule superset_rule(std::set<uidType> dependents, uidType uid) {
        std::pair<uidType, uidType> bounds;
        std::set<uidType> c_set, d_set;

        std::vector<Rule> dependent_rules;
        std::transform(dependents.begin(), dependents.end(), std::back_inserter(dependent_rules),
                       [this](uidType id) -> Rule { return this->of(id); });

        // get min lower bound and max upper bound
        uidType min_lower = uidType::MIN;
        for (uidType uid : uidType::range(0, uid)) {
            bool is_lower_bound = true;
            for (const auto rule : dependent_rules) {
                is_lower_bound = is_lower_bound && uid_le(uid, rule.bounds.first);
            }
            if (is_lower_bound && uid_ge(uid, min_lower))
                min_lower = uid;
        }

        // std::cout << "uid = " << uid.str() << "\n";
        uidType max_upper = uidType::MAX;
        for (uidType uid : uidType::range(0, uid)) {
            bool is_upper_bound = true;
            for (const auto rule : dependent_rules) {
                is_upper_bound = is_upper_bound && uid_ge(uid, rule.bounds.second);
                // std::cout << "Compare " << uid.str() << " with " << rule.bounds.second.str() << ": " << is_upper_bound << "\n";
            }
            if (is_upper_bound && uid_le(uid, max_upper)) {
                max_upper = uid;
            }
        }

        // get intersections of c_sets and d_sets, respectively
        for (uidType uid : uidType::range(0, uid)) {
            bool in_c_set = true, in_d_set = true;

            for (const auto rule : dependent_rules) {
                in_c_set = in_c_set && (rule.connected.find(uid) != rule.connected.end());
                in_d_set = in_d_set && (rule.disconnected.find(uid) != rule.disconnected.end());
            }

            if (in_c_set)
                c_set.insert(uid);
            if (in_d_set)
                d_set.insert(uid);
        }

        return { std::make_pair(min_lower, max_upper), c_set, d_set };
    }

    RulesReuse reuse_frontier() const {
        RulesReuse reuse_rules;
        for (size_t i = 0; i < size(); ++i) {
            const auto &current = this->of(i+1);
            RuleReuse new_rule(current);
            // is there any previous frontier we could reuse?
            for (int pi = i - 1; pi >= 0; --pi) {
                if (this->is_subset(i + 1, pi + 1) && !this->is_rule_simple(pi+1)) {
                    const auto &prev = this->of(pi+1);
                    new_rule.reuse = std::optional(pi+1);
                    for (auto c : prev.connected)
                        new_rule.rule.connected.erase(c);
                    for (auto d : prev.disconnected)
                        new_rule.rule.disconnected.erase(d);
                    break;
                }
            }
            reuse_rules.push_back(new_rule);
        }

        return reuse_rules;
    }

};


// A guided rule for extending a vertex:
// guide uids are those which initiate the extension procedure: N(id1)-N(id2)
// Special cases:
// - empty id1: any uid in rule.connected could work as a guide
// - empty id2: we don't need a second guide uid
// - empty update_id: cmap won't be updated by this rule
enum DataSource { NONE, EDGELIST, FRONTIER, ANY };

static std::string
to_string(uidType id, DataSource src) {
    std::stringstream ss;
    switch (src) {
        case EDGELIST:
            ss << "N(" << id.str() << ")";
            break;
        case FRONTIER:
            ss << "F(" << id.str() << ")";
            break;
        case ANY:
            ss << "N(<any>)";
            break;
        default:
            ss << "<null>";
            break;
    }
    return ss.str();
}

struct Guide {
    uidType id1, id2;
    DataSource src1, src2;
    std::optional<uidType> update_id;

    std::string str() const {
        std::stringstream ss;

        ss << to_string(id1, src1) << "-" << to_string(id2, src2);
        if (update_id)
            ss << " && update(" << update_id.value().str() << ")";

        return ss.str();
    }
};

class GuidedRules {
private:
    std::vector<Guide>  guides;
    Rules rules;

public:

    Guide guide_of(uidType uid) const {
        return guides.at(uid.value()-1);
    }

    Rule rule_of(uidType uid) const {
        return rules.of(uid);
    }

    // Attempt to add "guide" uids to each rule
    GuidedRules (const Rules &rules): rules(rules) {

        auto support_map = rules.get_support_map();

        // uids cached in cmap
        std::set<uidType> cmap_uids;

        for (size_t i = 0; i < rules.size(); ++i) {
            // level i+1

            // connected & disconnected uids
            auto &c_uids = rules[i].connected;
            auto &d_uids = rules[i].disconnected;

            std::vector<uidType> dependent_uids;
            std::set_union(c_uids.begin(), c_uids.end(),
                           d_uids.begin(), d_uids.end(),
                           std::back_inserter(dependent_uids));

            // uids that will be reused in some level > i+1
            std::vector<uidType> range = uidType::range(0, i+1);
            std::set<uidType> reuse_uids;
            std::copy_if(range.begin(), range.end(), std::inserter(reuse_uids, reuse_uids.end()),
                         [&support_map, i](const uidType &uid) {
                             return support_map[uid].upper_bound(i+1) != support_map[uid].end();
                         });

            // don't-care uids
            std::vector<uidType> dc_uids;
            std::set_difference(range.begin(), range.end(),
                                dependent_uids.begin(), dependent_uids.end(),
                                std::back_inserter(dc_uids));
            // don't-care uids that will be used in some level > i+1
            std::vector<uidType> r_uids;
            std::set_intersection(dc_uids.begin(), dc_uids.end(),
                                  reuse_uids.begin(), reuse_uids.end(),
                                  std::back_inserter(r_uids));

            // those connected, disconnected & don't-care uids uncached in the cmap
            std::vector<uidType> c_uncmapped;
            std::vector<uidType> d_uncmapped;
            std::vector<uidType> r_uncmapped;
            std::copy_if(c_uids.begin(), c_uids.end(), std::back_inserter(c_uncmapped),
                         [&](const uidType &uid) { return cmap_uids.find(uid) == cmap_uids.end(); });
            std::copy_if(d_uids.begin(), d_uids.end(), std::back_inserter(d_uncmapped),
                         [&](const uidType &uid) { return cmap_uids.find(uid) == cmap_uids.end(); });
            std::copy_if(r_uids.begin(), r_uids.end(), std::back_inserter(r_uncmapped),
                         [&](const uidType &uid) { return cmap_uids.find(uid) == cmap_uids.end(); });

            // invariant: there's at most one uid in [0, i+1) not cached in cmap
            assert(c_uncmapped.size() + d_uncmapped.size() + r_uncmapped.size() <= 1);

            uidType id1, id2;
            DataSource src1, src2;
            std::optional<uidType> update_id;
            // std::optional<uidType> guide_id1, guide_id2, update_id;

            if (c_uncmapped.size() + d_uncmapped.size() + r_uncmapped.size() == 0) {
                src1 = ANY;
                src2 = NONE;
                update_id = std::nullopt;

            } else if (c_uncmapped.size() == 1) {
                id1 = *c_uncmapped.begin();
                src1 = EDGELIST;
                src2 = NONE;
                update_id = (reuse_uids.find(id1) != reuse_uids.end()) ? std::optional(id1) : std::nullopt;

            } else if (d_uncmapped.size() == 1) {
                src1 = ANY;
                id2 = *d_uncmapped.begin();
                src2 = EDGELIST;
                update_id = (reuse_uids.find(id2) != reuse_uids.end()) ? std::optional(id2) : std::nullopt;
            } else {
                src1 = ANY;
                src2 = NONE;
                auto uid = *r_uncmapped.begin();
                update_id = std::optional(uid);
            }

            guides.push_back({ id1, id2, src1, src2, update_id });
            if (update_id) {
                cmap_uids.insert(update_id.value());
            }
        }
    }

    // Attemp to determine the uid with ANY src in guides
    // @param   reuse_frontier  flag to do frontier reuse optimization
    // @return  new GuidedRules obj with src of guide uid determined
    GuidedRules assign_sources(bool reuse_frontier = true) {

        GuidedRules other(*this);

        for (size_t i = 0; i < guides.size(); ++i) {
            auto &current = other.guides[i];

            if (current.src1 == ANY) {
                bool found_frontier = false;
                if (reuse_frontier) {
                    // is there any previous frontier we could reuse?
                    for (int pi = i - 1; pi >= 0; --pi) {
                        if (rules.is_subset(i + 1, pi + 1) && !rules.is_rule_simple(pi+1)) {
                            current.id1 = pi + 1;
                            current.src1 = FRONTIER;
                            found_frontier = true;
                            break;
                        }
                    }
                }
                if (!found_frontier) {
                    // no frontier reusable, select a uid from c-set
                    current.id1 = *(rules[i].connected.begin());
                    current.src1 = EDGELIST;
                }
            }

            assert (current.src2 != ANY && "src2 cannot be ANY");
        }

        return other;
    }

    GuidedRules simplify() {

        GuidedRules other(*this);

        // For each guide and rule,
        // remove id1 and id2 from the connected & disconnected set
        // because they are by default (dis)connected
        for (size_t i = 0; i < guides.size(); ++i) {
            auto &current = other.guides[i];

            auto &c_set = other.rules[i].connected;
            if (current.src1 == EDGELIST) {
                c_set.erase(current.id1);
            } else if (current.src1 == FRONTIER) {
                // remove all pre-connected uids
                auto &c_set_prev = rules.of(current.id1).connected;
                for (auto uid : c_set_prev) {
                    c_set.erase(uid);
                }
            }

            auto &d_set = other.rules[i].disconnected;
            if (current.src2 == EDGELIST) {
                d_set.erase(current.id2);
            } else if (current.src2 == FRONTIER) {
                // remove all pre-d_set uids
                auto &d_set_prev = rules.of(current.id2).disconnected;
                for (auto uid : d_set_prev) {
                    d_set.erase(uid);
                }
            }
        }

        // remove redundant update_id
        auto support_map = other.rules.get_support_map();

        for (size_t i = 0; i < guides.size(); ++i) {
            auto &guide = other.guides[i];
            if (guide.update_id) {
                uidType uid = guide.update_id.value();
                if (support_map[uid].upper_bound(i + 1) == support_map[uid].end()) {
                    guide.update_id = std::nullopt;
                }
            }
        }

        return other;
    }

    std::vector<std::string> strs() {
        std::vector<std::string> vec;
        for (size_t i = 0; i < guides.size(); ++i) {
            std::stringstream ss;
            ss <<  "{ " << guides[i].str()  << " pruneBy " << rules[i].str();
            vec.push_back(ss.str());
        }
        return vec;
    }

};



// constructs for EncodedRule:
// LookupPolicy && UpdatePolicy

template <typename VT>
struct LookupPolicy {
    VT mask_bits;       // masks out vertices we don't care
    VT cond_bits;       // connectivity bits

    inline bool cond(VT bucket_bits) const {
        return (bucket_bits & mask_bits) == cond_bits;
    }

    std::string str() const {
        constexpr size_t BITS = sizeof(VT) * 8;
        std::stringstream ss;
        ss << "mask=" << std::bitset<BITS>(mask_bits) << ", "
           << "cond=" << std::bitset<BITS>(cond_bits);
        return ss.str();
    }
};

template <typename VT>
struct UpdatePolicy {
    VT mask_bits;       // masks out vertices we don't care
    VT cond_bits;       // connectivity bits
    VT upd_bits;        // usually only one bit is set

    inline bool update_cond(VT bucket_bits) const {
        return (bucket_bits & mask_bits) == cond_bits;
    }

    inline VT update_op(VT bucket_bits) const {
        return bucket_bits | upd_bits;
    }

    inline bool restore_cond(VT bucket_bits) const {
        return (bucket_bits & (mask_bits | upd_bits)) ==
               (cond_bits | upd_bits);
    }


    inline VT restore_op(VT bucket_bits) const {
        return bucket_bits & (~upd_bits);
    }

    std::string str() const {
        constexpr size_t BITS = sizeof(VT) * 8;
        std::stringstream ss;
        ss << "mask=" << std::bitset<BITS>(mask_bits) << ", "
           << "cond=" << std::bitset<BITS>(cond_bits) << ", "
           << "upd=" << std::bitset<BITS>(upd_bits);
        return ss.str();
    }
};

struct LookupRule {
    vidType vid1, vid2;
    DataSource src1, src2;
    vidType lower, upper;
    LookupPolicy<uint8_t> policy;
};

struct UpdateRule {
    bool enable;
    vidType vid;
    vidType lower, upper;
    UpdatePolicy<uint8_t> policy;
};

using RestoreRule = UpdateRule;

struct EncodedRule {
    uidType id1, id2;
    DataSource src1, src2;
    uidType lookup_lower, lookup_upper;
    LookupPolicy<uint8_t> lookup_policy;

    bool update_enable;
    uidType update_id;
    uidType update_lower, update_upper;
    UpdatePolicy<uint8_t> update_policy;

    LookupRule lookup_with(const std::vector<vidType> &history) const {
        return {
            history.at(id1.value()),
            history.at(id2.value()),
            src1, src2,
            lookup_lower.is_bounded()? history.at(lookup_lower.value()) : VID_MIN,
            lookup_upper.is_bounded()? history.at(lookup_upper.value()) : VID_MAX,
            lookup_policy
        };
    }

    UpdateRule update_with(const std::vector<vidType> &history) const {
        return {
            update_enable,
            history.at(update_id.value()),
            update_lower.is_bounded()? history.at(update_lower.value()) : VID_MIN,
            update_upper.is_bounded()? history.at(update_upper.value()) : VID_MAX,
            update_policy
        };
    }

    RestoreRule restore_with(const std::vector<vidType> &history) const {
        return {
            update_enable,
            history.at(update_id.value()),
            update_lower.is_bounded()? history.at(update_lower.value()) : VID_MIN,
            update_upper.is_bounded()? history.at(update_upper.value()) : VID_MAX,
            update_policy
        };
    }

    std::string lookup_str() const {
        std::stringstream ss;
        ss << to_string(id1, src1) << "-" << to_string(id2, src2) << ", ";
        ss << "range=[" << lookup_lower.str() << "," << lookup_upper.str() << "), ";
        ss << lookup_policy.str();
        return ss.str();
    }

    std::string update_str() const {
        if (!update_enable) {
            return "no-update";
        }
        std::stringstream ss;
        ss << "update=" << to_string(update_id, EDGELIST) << ", ";
        ss << "range=[" << update_lower.str() << "," << update_upper.str() << "), ";
        ss << update_policy.str();
        return ss.str();
    }
};

using EncodedRules = std::vector<EncodedRule>;

// an execution plan for matching a specific pattern;
// will be passed to a graph pattern mining template hardware / software
class ExecutionPlan {
private:
    // rules.size() + 1 equals the pattern size
    // uid in rules[i].connected and rules[i].disconnected <= i
    Rules rules;
    RulesReuse reuse_rules;
    EncodedRules encoded;

    static LookupPolicy<uint8_t>
    encode(const std::set<uidType> &c_set, const std::set<uidType> &d_set) {
        uint8_t mask = 0;
        uint8_t cond = 0;
        for (auto id : c_set) {
            mask |= (1 << id.value());
            cond |= (1 << id.value());
        }
        for (auto id : d_set) {
            mask |= (1 << id.value());
        }
        return {mask, cond};
    }

    static UpdatePolicy<uint8_t>
    encode(const std::set<uidType> &c_set, const std::set<uidType> &d_set, uidType uid) {
        uint8_t mask = 0;
        uint8_t cond = 0;
        uint8_t upd = 1 << uid.value();
        for (auto id : c_set) {
            mask |= (1 << id.value());
            cond |= (1 << id.value());
        }
        for (auto id : d_set) {
            mask |= (1 << id.value());
        }
        return {mask, cond, upd};
    }

public:

    ExecutionPlan() = default;

    // @param   vec_rules           a sequence of extension steps for a specific pattern
    // @param   difference_capable  true if the hardware provides an extra difference engine
    //                              to accelerate the extension
    // constructs an execution plan (a sequence of encoded rules) for the sw/hw template
    // TODO: handle the difference_capable flag
    ExecutionPlan(std::vector<Rule> vec_rules,
                  bool reuse_frontier = true,
                  bool difference_capable = true):
        rules(vec_rules),
        reuse_rules(rules.reuse_frontier())
    {
        GuidedRules undetermined(rules);

        GuidedRules simplified = undetermined.assign_sources(reuse_frontier)
                                             .simplify();

        // then encode GuidedRules to EncodedRules, and try minimizing cmap footage

        auto support_map = rules.get_support_map();
        std::set<uidType> cmap_uids;

        for (size_t i = 0; i < rules.size(); ++i) {
            // encode the rule of i+1
            uidType uid = i + 1;
            EncodedRule e_rule;
            const auto guide = simplified.guide_of(uid);
            const auto s_rule = simplified.rule_of(uid);

            e_rule.id1 = guide.id1; e_rule.src1 = guide.src1;
            e_rule.id2 = guide.id2; e_rule.src2 = guide.src2;
            e_rule.lookup_lower = s_rule.bounds.first;
            e_rule.lookup_upper = s_rule.bounds.second;
            e_rule.lookup_policy = encode(s_rule.connected, s_rule.disconnected);

            e_rule.update_enable = guide.update_id.has_value();
            // generate the policy to update cmap
            if (e_rule.update_enable) {
                e_rule.update_id = guide.update_id.value();

                // try to minimize the cmap footage by finding out
                // a "superset" rule for cmap update
                auto dependents = support_map.at(e_rule.update_id);
                Rule superset = rules.superset_rule(dependents, uid);

                // only those uids which are already in cmap will work
                std::set<uidType> c_set;
                for (auto uid : superset.connected) {
                    if (cmap_uids.find(uid) != cmap_uids.end())
                        c_set.insert(uid);
                }
                std::set<uidType> d_set = superset.disconnected;

                e_rule.update_lower = superset.bounds.first;
                e_rule.update_upper = superset.bounds.second;
                e_rule.update_policy = encode(c_set, d_set, e_rule.update_id);

                cmap_uids.insert(e_rule.update_id);
            }

            encoded.push_back(e_rule);
        }

    }

    size_t pattern_size() const { return rules.size() + 1; }

    // @param level     >=1, the level of encoded rule
    const EncodedRule &at(size_t level) const {
        return encoded.at(level-1);
    }

    const Rule &rule_at(size_t level) const {
        return rules.of(level);
    }

    const RuleReuse &reuse_at(size_t level) const {
        return reuse_rules.at(level-1);
    }

    std::string rules_str() const {
        std::stringstream ss;
        ss << "==================\n";
        for (size_t i = 0; i < rules.size(); ++i) {
            ss << rules[i].str() << "\n";
        }
        ss << "==================\n";
        return ss.str();
    }

    std::string reuse_str() const {
        std::stringstream ss;
        ss << "==================\n";
        for (size_t i = 0; i < reuse_rules.size(); ++i) {
            ss << reuse_rules[i].str() << "\n";
        }
        ss << "==================\n";
        return ss.str();
    }

    std::string str() const {
        std::stringstream ss;
        ss << "==================\n";
        ss << "u0 := V\n";
        uidType uid = 1;
        for (auto r : encoded) {
            ss << uid.str() << " := " << r.lookup_str() << "\n";
            ss << "      " << r.update_str() << "\n";
            uid = uid.value() + 1;
        }
        ss << "==================\n";
        return ss.str();
    }
};

