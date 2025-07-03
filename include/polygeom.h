#pragma once

#include "common.h"
#include <vector>

namespace srpgt {

namespace polygeom {

// Basic polygon operations
double polygon_area(const Polygon& poly);
double polygon_signed_area(const Polygon& poly);
bool is_polygon_clockwise(const Polygon& poly);
bool point_in_polygon(const Point2D& point, const Polygon& poly);

// Distance functions
double point_to_polygon_distance(const Point2D& point, const Polygon& poly);
std::vector<double> points_to_polygon_distances(const std::vector<Point2D>& points, const Polygon& poly);

// Polygon intersection with line
std::vector<Point2D> polygon_line_intersection(const Polygon& poly, const Point2D& line_point, const Vector2D& line_dir);

// Convex hull computation
Polygon convex_hull(const std::vector<Point2D>& points);

// Polygon erosion/dilation
Polygon erode_convex_polygon(const Polygon& poly, double radius);

// Line intersection
Point2D line_intersection(const Point2D& p1, const Vector2D& dir1, 
                         const Point2D& p2, const Vector2D& dir2);

// Check if three points are collinear
bool collinear(const Point2D& a, const Point2D& b, const Point2D& c, double threshold = EPS);

} // namespace polygeom

} // namespace srpgt