#pragma once

#include "common.h"
#include <vector>
#include <unordered_map>

namespace srpgt {

class DisjointSet {
public:
    DisjointSet();
    explicit DisjointSet(size_t n);
    
    // Add a new element
    void add(size_t element);
    
    // Find the root of an element with path compression
    size_t find(size_t element);
    
    // Merge two sets
    void merge(size_t a, size_t b);
    
    // Check if two elements are connected
    bool connected(size_t a, size_t b);
    
    // Get the size of the set containing element
    size_t set_size(size_t element);
    
    // Get number of disjoint sets
    size_t num_sets() const;

private:
    std::unordered_map<size_t, size_t> parent_;
    std::unordered_map<size_t, size_t> rank_;
    std::unordered_map<size_t, size_t> size_;
    size_t num_sets_;
};

// Build disjoint sets from points using Manhattan distance
DisjointSet build_disjoint_sets(const std::vector<Point2D>& points, double threshold);

} // namespace srpgt