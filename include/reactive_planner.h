#pragma once

#include "common.h"
#include <vector>
#include <map>

namespace srpgt {

// Diffeo parameters structure
struct DiffeoParams {
    double p = 10.0;
    double epsilon = 4.0;
    double varepsilon = 4.0;
    double mu_1 = 1.0;      // beta switch parameter
    double mu_2 = 0.15;     // gamma switch parameter
    Polygon workspace;
};

// Triangle for diffeomorphism
struct Triangle {
    Point2D vertices[3];
    
    Triangle() = default;
    Triangle(const Point2D& a, const Point2D& b, const Point2D& c) {
        vertices[0] = a;
        vertices[1] = b;
        vertices[2] = c;
    }
};

// Tree node for diffeo decomposition
struct DiffeoTreeNode {
    Polygon vertices;
    int predecessor = -1;
    int depth = 0;
    std::vector<int> children;
};

using DiffeoTree = std::vector<DiffeoTreeNode>;

namespace reactive_planner {

// Core diffeomorphism functions
Point2D triangle_diffeo(const Point2D& position, const Triangle& triangle, 
                       const DiffeoParams& params);

Point2D polygon_diffeo(const Point2D& position, const Polygon& polygon, 
                      const DiffeoParams& params);

// Switching functions
double triangle_switch(const Point2D& position, const Triangle& triangle, 
                      const DiffeoParams& params);

double polygon_switch(const Point2D& position, const Polygon& polygon, 
                     const DiffeoParams& params);

// Tree-based diffeomorphism
Point2D polygon_diffeo_triangulation(const Point2D& position, 
                                     const DiffeoTree& tree, 
                                     const DiffeoParams& params);

// Build diffeo tree from polygon
DiffeoTree build_diffeo_tree_triangulation(const Polygon& polygon_vertices, 
                                          const DiffeoParams& params);

// Implicit function evaluations
double outside_implicit_triangle(const Point2D& position, const Triangle& triangle, 
                               const DiffeoParams& params);

double inside_implicit_triangle(const Point2D& position, const Triangle& triangle, 
                              const DiffeoParams& params);

} // namespace reactive_planner

} // namespace srpgt