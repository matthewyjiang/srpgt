#include "robot.h"
#include "polygeom.h"
#include "reactive_planner.h"
#include "disjoint_set.h"
#include "config.h"
#include "gaussian_optimization.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace srpgt;

void test_point2d() {
    std::cout << "Testing Point2D..." << std::endl;
    
    Point2D p1(3, 4);
    Point2D p2(1, 1);
    
    Point2D sum = p1 + p2;
    assert(std::abs(sum.x() - 4.0) < EPS);
    assert(std::abs(sum.y() - 5.0) < EPS);
    
    Point2D diff = p1 - p2;
    assert(std::abs(diff.x() - 2.0) < EPS);
    assert(std::abs(diff.y() - 3.0) < EPS);
    
    double norm = p1.norm();
    assert(std::abs(norm - 5.0) < EPS);
    
    std::cout << "Point2D tests passed!" << std::endl;
}

void test_robot() {
    std::cout << "Testing Robot..." << std::endl;
    
    Robot robot(10, 20, 2.0);
    
    // Test initial position
    assert(std::abs(robot.position().x() - 10.0) < EPS);
    assert(std::abs(robot.position().y() - 20.0) < EPS);
    assert(std::abs(robot.radius() - 2.0) < EPS);
    
    // Test movement
    Vector2D velocity(1, 0);
    robot.update(velocity);
    
    // Position should have changed
    assert(robot.position().x() > 10.0);
    assert(robot.trail().size() > 0);
    
    std::cout << "Robot tests passed!" << std::endl;
}

void test_polygon_geometry() {
    std::cout << "Testing polygon geometry..." << std::endl;
    
    // Test triangle area
    Polygon triangle = {
        Point2D(0, 0),
        Point2D(1, 0),
        Point2D(0, 1)
    };
    
    double area = polygeom::polygon_area(triangle);
    assert(std::abs(area - 0.5) < EPS);
    
    // Test point in polygon
    Point2D inside(0.25, 0.25);
    Point2D outside(1, 1);
    
    assert(polygeom::point_in_polygon(inside, triangle));
    assert(!polygeom::point_in_polygon(outside, triangle));
    
    std::cout << "Polygon geometry tests passed!" << std::endl;
}

void test_disjoint_set() {
    std::cout << "Testing disjoint set..." << std::endl;
    
    std::vector<Point2D> points = {
        Point2D(0, 0),
        Point2D(1, 0),  // Close to first point
        Point2D(10, 10), // Far from others
        Point2D(11, 10)  // Close to third point
    };
    
    DisjointSet ds = build_disjoint_sets(points, 2.0);
    
    // Points 0 and 1 should be connected
    assert(ds.connected(0, 1));
    
    // Points 2 and 3 should be connected
    assert(ds.connected(2, 3));
    
    // Points 0 and 2 should not be connected
    assert(!ds.connected(0, 2));
    
    std::cout << "Disjoint set tests passed!" << std::endl;
}

void test_config() {
    std::cout << "Testing configuration..." << std::endl;
    
    Config config;
    
    // Test default values
    double robot_radius = config.get_double("robot", "ROBOT_RADIUS", 1.0);
    assert(std::abs(robot_radius - 2.0) < EPS);
    
    // Test setting values
    config.set_double("test", "value", 42.5);
    double test_value = config.get_double("test", "value", 0.0);
    assert(std::abs(test_value - 42.5) < EPS);
    
    std::cout << "Configuration tests passed!" << std::endl;
}

void test_reactive_planner() {
    std::cout << "Testing reactive planner..." << std::endl;
    
    Triangle triangle(Point2D(0, 0), Point2D(1, 0), Point2D(0, 1));
    DiffeoParams params;
    
    Point2D position(0.5, 0.5);
    
    // Test triangle switch function
    double switch_val = reactive_planner::triangle_switch(position, triangle, params);
    assert(switch_val >= 0.0 && switch_val <= 1.0);
    
    // Test triangle diffeomorphism
    Point2D transformed = reactive_planner::triangle_diffeo(position, triangle, params);
    // Should return a valid point
    assert(!std::isnan(transformed.x()) && !std::isnan(transformed.y()));
    
    std::cout << "Reactive planner tests passed!" << std::endl;
}

void test_gaussian_optimization() {
    std::cout << "Testing Gaussian optimization..." << std::endl;
    
    GPOptimizationParams params;
    params.kernel_variance = 1.0;
    params.kernel_lengthscale = 1.0;
    params.beta = 2.0;
    params.threshold = 0.0;
    
    GaussianOptimizer optimizer(params);
    
    // Set parameter bounds
    optimizer.set_parameter_bounds(0, 10, 0, 10, 10);
    
    // Create some training data
    std::vector<Point2D> training_points = {
        Point2D(1, 1), Point2D(2, 2), Point2D(3, 3)
    };
    std::vector<double> training_values = {1.0, 2.0, 3.0};
    
    optimizer.initialize(training_points, training_values);
    
    // Test prediction
    auto prediction = optimizer.predict(Point2D(2.5, 2.5));
    assert(!std::isnan(prediction.first));
    assert(!std::isnan(prediction.second));
    assert(prediction.second > 0); // Uncertainty should be positive
    
    // Test getting next sample
    Point2D goal(5, 5);
    Point2D next_sample = optimizer.get_next_sample(goal);
    assert(!std::isnan(next_sample.x()) && !std::isnan(next_sample.y()));
    
    // Test adding observation
    optimizer.add_observation(Point2D(4, 4), 4.0);
    
    // Get safe parameters
    auto safe_params = optimizer.get_safe_parameters();
    assert(!safe_params.empty());
    
    std::cout << "Gaussian optimization tests passed!" << std::endl;
}

int main() {
    std::cout << "Running SRPGT C++ Tests" << std::endl;
    std::cout << "=======================" << std::endl;
    
    try {
        test_point2d();
        test_robot();
        test_polygon_geometry();
        test_disjoint_set();
        test_config();
        test_reactive_planner();
        test_gaussian_optimization();
        
        std::cout << "\nAll tests passed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}