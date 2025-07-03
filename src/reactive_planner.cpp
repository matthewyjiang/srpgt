#include "reactive_planner.h"
#include "polygeom.h"
#include <cmath>
#include <algorithm>

namespace srpgt {
namespace reactive_planner {

Point2D triangle_diffeo(const Point2D& position, const Triangle& triangle, 
                       const DiffeoParams& params) {
    // Simplified version of the triangle diffeomorphism
    // This is a basic implementation - the full algorithm is quite complex
    
    // For now, return a simple transformation
    Point2D center = (triangle.vertices[0] + triangle.vertices[1] + triangle.vertices[2]) * (1.0/3.0);
    Vector2D to_center = center - position;
    
    // Apply a simple radial transformation
    double dist = to_center.norm();
    if (dist < EPS) return position;
    
    double factor = 1.0 / (1.0 + params.epsilon * std::exp(-dist / params.varepsilon));
    return position + to_center * factor;
}

Point2D polygon_diffeo(const Point2D& position, const Polygon& polygon, 
                      const DiffeoParams& params) {
    if (polygon.empty()) return position;
    
    // Simple implementation: move towards polygon centroid
    Point2D centroid(0, 0);
    for (const auto& vertex : polygon) {
        centroid += vertex;
    }
    centroid = centroid * (1.0 / polygon.size());
    
    Vector2D to_centroid = centroid - position;
    double dist = to_centroid.norm();
    
    if (dist < EPS) return position;
    
    double factor = 1.0 / (1.0 + params.epsilon * std::exp(-dist / params.varepsilon));
    return position + to_centroid * factor;
}

double triangle_switch(const Point2D& position, const Triangle& triangle, 
                      const DiffeoParams& params) {
    // Beta switch function for triangles
    double min_dist = std::numeric_limits<double>::max();
    
    // Find minimum distance to triangle edges
    for (int i = 0; i < 3; ++i) {
        int j = (i + 1) % 3;
        Vector2D edge = triangle.vertices[j] - triangle.vertices[i];
        Vector2D to_point = position - triangle.vertices[i];
        
        double edge_length_sq = edge.x() * edge.x() + edge.y() * edge.y();
        if (edge_length_sq < EPS) continue;
        
        double t = dot(to_point, edge) / edge_length_sq;
        t = std::max(0.0, std::min(1.0, t));
        
        Point2D closest = triangle.vertices[i] + edge * t;
        min_dist = std::min(min_dist, distance(position, closest));
    }
    
    // Apply switch function
    return 1.0 / (1.0 + std::exp(-params.mu_1 * (min_dist - params.mu_2)));
}

double polygon_switch(const Point2D& position, const Polygon& polygon, 
                     const DiffeoParams& params) {
    if (polygon.empty()) return 0.0;
    
    double min_dist = polygeom::point_to_polygon_distance(position, polygon);
    
    // Apply switch function
    return 1.0 / (1.0 + std::exp(-params.mu_1 * (min_dist - params.mu_2)));
}

Point2D polygon_diffeo_triangulation(const Point2D& position, 
                                     const DiffeoTree& tree, 
                                     const DiffeoParams& params) {
    if (tree.empty()) return position;
    
    Point2D result = position;
    
    // Apply diffeomorphism from each node in the tree
    for (const auto& node : tree) {
        if (node.vertices.size() >= 3) {
            // Convert polygon to triangle for simplicity
            Triangle tri(node.vertices[0], node.vertices[1], node.vertices[2]);
            
            double switch_val = triangle_switch(result, tri, params);
            Point2D diffeo_result = triangle_diffeo(result, tri, params);
            
            // Blend based on switch function
            result = result * (1.0 - switch_val) + diffeo_result * switch_val;
        }
    }
    
    return result;
}

DiffeoTree build_diffeo_tree_triangulation(const Polygon& polygon_vertices, 
                                          const DiffeoParams& params) {
    DiffeoTree tree;
    
    if (polygon_vertices.size() < 3) return tree;
    
    // Simple triangulation: fan triangulation from first vertex
    for (size_t i = 1; i + 1 < polygon_vertices.size(); ++i) {
        DiffeoTreeNode node;
        node.vertices = {polygon_vertices[0], polygon_vertices[i], polygon_vertices[i+1]};
        node.predecessor = -1;
        node.depth = 0;
        tree.push_back(node);
    }
    
    return tree;
}

double outside_implicit_triangle(const Point2D& position, const Triangle& triangle, 
                               const DiffeoParams& params) {
    // Simplified implicit function
    double min_dist = std::numeric_limits<double>::max();
    
    for (int i = 0; i < 3; ++i) {
        min_dist = std::min(min_dist, distance(position, triangle.vertices[i]));
    }
    
    return std::pow(min_dist, params.p);
}

double inside_implicit_triangle(const Point2D& position, const Triangle& triangle, 
                              const DiffeoParams& params) {
    // Check if point is inside triangle
    Vector2D v0 = triangle.vertices[2] - triangle.vertices[0];
    Vector2D v1 = triangle.vertices[1] - triangle.vertices[0];
    Vector2D v2 = position - triangle.vertices[0];
    
    double dot00 = dot(v0, v0);
    double dot01 = dot(v0, v1);
    double dot02 = dot(v0, v2);
    double dot11 = dot(v1, v1);
    double dot12 = dot(v1, v2);
    
    double inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    double u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    double v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
    
    bool inside = (u >= 0) && (v >= 0) && (u + v <= 1);
    
    if (inside) {
        return 1.0;
    } else {
        return outside_implicit_triangle(position, triangle, params);
    }
}

} // namespace reactive_planner
} // namespace srpgt