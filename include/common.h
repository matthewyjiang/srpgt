#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <memory>
#include <Eigen/Dense>

namespace srpgt {

// Use Eigen for 2D vectors and points
using Point2D = Eigen::Vector2d;
using Vector2D = Eigen::Vector2d;

// Polygon representation
using Polygon = std::vector<Point2D>;

// Common mathematical constants
constexpr double PI = 3.14159265358979323846;
constexpr double EPS = 1e-9;

// Utility functions
inline double distance(const Point2D& a, const Point2D& b) {
    return (a - b).norm();
}

inline double dot(const Vector2D& a, const Vector2D& b) {
    return a.dot(b);
}

inline double cross(const Vector2D& a, const Vector2D& b) {
    return a.x() * b.y() - a.y() * b.x();
}

} // namespace srpgt