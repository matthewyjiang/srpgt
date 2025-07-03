#include "polygeom.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace srpgt {
namespace polygeom {

double polygon_area(const Polygon& poly) {
    return std::abs(polygon_signed_area(poly));
}

double polygon_signed_area(const Polygon& poly) {
    if (poly.size() < 3) return 0.0;
    
    double area = 0.0;
    size_t n = poly.size();
    
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        area += poly[i].x * poly[j].y - poly[j].x * poly[i].y;
    }
    
    return area * 0.5;
}

bool is_polygon_clockwise(const Polygon& poly) {
    return polygon_signed_area(poly) < 0.0;
}

bool point_in_polygon(const Point2D& point, const Polygon& poly) {
    if (poly.size() < 3) return false;
    
    bool inside = false;
    size_t n = poly.size();
    
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        if (((poly[i].y > point.y) != (poly[j].y > point.y)) &&
            (point.x < (poly[j].x - poly[i].x) * (point.y - poly[i].y) / (poly[j].y - poly[i].y) + poly[i].x)) {
            inside = !inside;
        }
    }
    
    return inside;
}

double point_to_polygon_distance(const Point2D& point, const Polygon& poly) {
    if (poly.empty()) return std::numeric_limits<double>::infinity();
    
    double min_dist = std::numeric_limits<double>::infinity();
    
    // Check distance to each edge
    for (size_t i = 0; i < poly.size(); ++i) {
        size_t j = (i + 1) % poly.size();
        
        Vector2D edge = poly[j] - poly[i];
        Vector2D to_point = point - poly[i];
        
        double edge_length_sq = edge.x * edge.x + edge.y * edge.y;
        if (edge_length_sq < EPS) {
            // Degenerate edge, just point distance
            min_dist = std::min(min_dist, distance(point, poly[i]));
            continue;
        }
        
        double t = dot(to_point, edge) / edge_length_sq;
        t = std::max(0.0, std::min(1.0, t)); // Clamp to edge
        
        Point2D closest = poly[i] + edge * t;
        min_dist = std::min(min_dist, distance(point, closest));
    }
    
    return min_dist;
}

std::vector<double> points_to_polygon_distances(const std::vector<Point2D>& points, const Polygon& poly) {
    std::vector<double> distances;
    distances.reserve(points.size());
    
    for (const auto& point : points) {
        distances.push_back(point_to_polygon_distance(point, poly));
    }
    
    return distances;
}

Point2D line_intersection(const Point2D& p1, const Vector2D& dir1, 
                         const Point2D& p2, const Vector2D& dir2) {
    double det = cross(dir1, dir2);
    
    if (std::abs(det) < EPS) {
        // Lines are parallel
        return Point2D(std::numeric_limits<double>::infinity(), 
                      std::numeric_limits<double>::infinity());
    }
    
    Vector2D diff = p2 - p1;
    double t = cross(diff, dir2) / det;
    
    return p1 + dir1 * t;
}

bool collinear(const Point2D& a, const Point2D& b, const Point2D& c, double threshold) {
    Vector2D ab = b - a;
    Vector2D ac = c - a;
    
    double cross_product = std::abs(cross(ab, ac));
    double ab_norm = ab.norm();
    double ac_norm = ac.norm();
    
    if (ab_norm < EPS || ac_norm < EPS) return true;
    
    double sin_angle = cross_product / (ab_norm * ac_norm);
    return sin_angle < threshold;
}

Polygon convex_hull(const std::vector<Point2D>& points) {
    if (points.size() < 3) return Polygon(points);
    
    // Graham scan algorithm
    std::vector<Point2D> sorted_points = points;
    
    // Find bottom-most point (and leftmost in case of tie)
    size_t min_idx = 0;
    for (size_t i = 1; i < sorted_points.size(); ++i) {
        if (sorted_points[i].y < sorted_points[min_idx].y ||
            (sorted_points[i].y == sorted_points[min_idx].y && sorted_points[i].x < sorted_points[min_idx].x)) {
            min_idx = i;
        }
    }
    std::swap(sorted_points[0], sorted_points[min_idx]);
    
    Point2D pivot = sorted_points[0];
    
    // Sort points by polar angle with respect to pivot
    std::sort(sorted_points.begin() + 1, sorted_points.end(),
              [&pivot](const Point2D& a, const Point2D& b) {
                  Vector2D va = a - pivot;
                  Vector2D vb = b - pivot;
                  double cross_product = cross(va, vb);
                  if (std::abs(cross_product) < EPS) {
                      // Collinear points, sort by distance
                      return va.norm() < vb.norm();
                  }
                  return cross_product > 0;
              });
    
    // Build convex hull
    Polygon hull;
    for (const auto& point : sorted_points) {
        while (hull.size() >= 2) {
            Vector2D v1 = hull[hull.size()-1] - hull[hull.size()-2];
            Vector2D v2 = point - hull[hull.size()-1];
            if (cross(v1, v2) <= 0) {
                hull.pop_back();
            } else {
                break;
            }
        }
        hull.push_back(point);
    }
    
    return hull;
}

Polygon erode_convex_polygon(const Polygon& poly, double radius) {
    if (poly.size() < 3 || radius <= 0) return poly;
    
    Polygon eroded;
    
    for (size_t i = 0; i < poly.size(); ++i) {
        size_t prev = (i - 1 + poly.size()) % poly.size();
        size_t next = (i + 1) % poly.size();
        
        Vector2D edge1 = poly[i] - poly[prev];
        Vector2D edge2 = poly[next] - poly[i];
        
        // Normalize edges
        edge1 = edge1.normalized();
        edge2 = edge2.normalized();
        
        // Calculate inward normals
        Vector2D normal1(-edge1.y, edge1.x);
        Vector2D normal2(-edge2.y, edge2.x);
        
        // Move vertex inward by radius along bisector
        Vector2D bisector = (normal1 + normal2).normalized();
        double sin_half_angle = cross(normal1, bisector);
        
        if (std::abs(sin_half_angle) > EPS) {
            double offset = radius / sin_half_angle;
            eroded.push_back(poly[i] + bisector * offset);
        }
    }
    
    return eroded;
}

} // namespace polygeom
} // namespace srpgt