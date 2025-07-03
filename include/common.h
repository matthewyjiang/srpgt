#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <memory>

namespace srpgt {

// Basic 2D point structure
struct Point2D {
    double x, y;
    
    Point2D() : x(0.0), y(0.0) {}
    Point2D(double x_, double y_) : x(x_), y(y_) {}
    
    Point2D operator+(const Point2D& other) const {
        return Point2D(x + other.x, y + other.y);
    }
    
    Point2D operator-(const Point2D& other) const {
        return Point2D(x - other.x, y - other.y);
    }
    
    Point2D operator*(double scalar) const {
        return Point2D(x * scalar, y * scalar);
    }
    
    Point2D& operator+=(const Point2D& other) {
        x += other.x;
        y += other.y;
        return *this;
    }
    
    double norm() const {
        return std::sqrt(x * x + y * y);
    }
    
    Point2D normalized() const {
        double n = norm();
        return (n > 1e-10) ? Point2D(x / n, y / n) : Point2D(0, 0);
    }
};

// 2D vector typedef
using Vector2D = Point2D;

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
    return a.x * b.x + a.y * b.y;
}

inline double cross(const Vector2D& a, const Vector2D& b) {
    return a.x * b.y - a.y * b.x;
}

} // namespace srpgt