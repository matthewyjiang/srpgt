#include "disjoint_set.h"
#include <algorithm>

namespace srpgt {

DisjointSet::DisjointSet() : num_sets_(0) {}

DisjointSet::DisjointSet(size_t n) : num_sets_(0) {
    for (size_t i = 0; i < n; ++i) {
        add(i);
    }
}

void DisjointSet::add(size_t element) {
    if (parent_.find(element) == parent_.end()) {
        parent_[element] = element;
        rank_[element] = 0;
        size_[element] = 1;
        num_sets_++;
    }
}

size_t DisjointSet::find(size_t element) {
    if (parent_.find(element) == parent_.end()) {
        add(element);
    }
    
    if (parent_[element] != element) {
        parent_[element] = find(parent_[element]); // Path compression
    }
    return parent_[element];
}

void DisjointSet::merge(size_t a, size_t b) {
    size_t root_a = find(a);
    size_t root_b = find(b);
    
    if (root_a == root_b) return;
    
    // Union by rank
    if (rank_[root_a] < rank_[root_b]) {
        parent_[root_a] = root_b;
        size_[root_b] += size_[root_a];
    } else if (rank_[root_a] > rank_[root_b]) {
        parent_[root_b] = root_a;
        size_[root_a] += size_[root_b];
    } else {
        parent_[root_b] = root_a;
        size_[root_a] += size_[root_b];
        rank_[root_a]++;
    }
    
    num_sets_--;
}

bool DisjointSet::connected(size_t a, size_t b) {
    return find(a) == find(b);
}

size_t DisjointSet::set_size(size_t element) {
    return size_[find(element)];
}

size_t DisjointSet::num_sets() const {
    return num_sets_;
}

DisjointSet build_disjoint_sets(const std::vector<Point2D>& points, double threshold) {
    DisjointSet ds(points.size());
    
    // Simple O(n^2) approach for now - can be optimized with spatial data structures
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            double dist = distance(points[i], points[j]);
            if (dist <= threshold) {
                ds.merge(i, j);
            }
        }
    }
    
    return ds;
}

} // namespace srpgt