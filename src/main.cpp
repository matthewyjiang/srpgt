#include "robot.h"
#include "config.h"
#include "polygeom.h"
#include "reactive_planner.h"
#include "disjoint_set.h"
#include "gaussian_optimization.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

using namespace srpgt;

int main(int argc, char* argv[]) {
    std::cout << "SRPGT - Safe Reactive Navigation for Granular Terrain Exploration (C++)" << std::endl;
    std::cout << "========================================================================" << std::endl;
    
    // Load configuration
    Config config;
    if (argc > 1) {
        if (config.load_from_file(argv[1])) {
            std::cout << "Loaded configuration from: " << argv[1] << std::endl;
        } else {
            std::cout << "Failed to load configuration, using defaults" << std::endl;
        }
    } else {
        std::cout << "Using default configuration (no config file specified)" << std::endl;
    }
    
    // Initialize robot
    double robot_x = config.get_double("robot", "STARTING_X", 40.0);
    double robot_y = config.get_double("robot", "STARTING_Y", 221.0);
    double robot_radius = config.get_double("robot", "ROBOT_RADIUS", 2.0);
    
    Robot robot(robot_x, robot_y, robot_radius);
    
    // Set goal
    Point2D goal(config.get_double("robot", "GOAL_X", 185.0),
                 config.get_double("robot", "GOAL_Y", 360.0));
    
    std::cout << "Robot initialized at: (" << robot_x << ", " << robot_y << ")" << std::endl;
    std::cout << "Goal set to: (" << goal.x() << ", " << goal.y() << ")" << std::endl;
    
    // Create some test obstacles
    std::vector<Polygon> obstacles;
    
    // Simple rectangular obstacle
    Polygon obstacle1 = {
        Point2D(100, 100),
        Point2D(150, 100),
        Point2D(150, 150),
        Point2D(100, 150)
    };
    obstacles.push_back(obstacle1);
    
    // Triangular obstacle
    Polygon obstacle2 = {
        Point2D(200, 200),
        Point2D(250, 180),
        Point2D(220, 250)
    };
    obstacles.push_back(obstacle2);
    
    std::cout << "Created " << obstacles.size() << " test obstacles" << std::endl;
    
    // Set up diffeo parameters
    DiffeoParams diffeo_params;
    diffeo_params.p = 10.0;
    diffeo_params.epsilon = 4.0;
    diffeo_params.varepsilon = 4.0;
    diffeo_params.mu_1 = 1.0;
    diffeo_params.mu_2 = 0.15;
    
    // Create workspace boundary
    diffeo_params.workspace = {
        Point2D(0, 0),
        Point2D(400, 0),
        Point2D(400, 400),
        Point2D(0, 400)
    };
    
    std::cout << "Starting simulation..." << std::endl;
    std::cout << "Press Ctrl+C to stop" << std::endl;
    
    // Simulation loop
    int step = 0;
    const int max_steps = 1000;
    const double dt = 0.1;
    
    while (step < max_steps) {
        // Calculate distance to goal
        double dist_to_goal = distance(robot.position(), goal);
        
        if (dist_to_goal < robot.radius()) {
            std::cout << "Goal reached at step " << step << "!" << std::endl;
            break;
        }
        
        // Simple navigation: move towards goal with obstacle avoidance
        Vector2D to_goal = goal - robot.position();
        Vector2D velocity = to_goal.normalized();
        
        // Apply reactive planning (simplified)
        for (const auto& obstacle : obstacles) {
            if (!obstacle.empty()) {
                double dist_to_obstacle = polygeom::point_to_polygon_distance(robot.position(), obstacle);
                
                if (dist_to_obstacle < robot.radius() * 3) {
                    // Apply diffeomorphism for obstacle avoidance
                    Point2D transformed = reactive_planner::polygon_diffeo(robot.position(), obstacle, diffeo_params);
                    Vector2D avoidance = transformed - robot.position();
                    
                    // Blend avoidance with goal-seeking
                    double blend_factor = std::exp(-dist_to_obstacle);
                    velocity = velocity * (1.0 - blend_factor) + avoidance.normalized() * blend_factor;
                }
            }
        }
        
        // Update robot
        robot.update(velocity * dt);
        
        // Print status every 100 steps
        if (step % 100 == 0) {
            std::cout << "Step " << step << ": " << robot.to_string() 
                      << ", dist_to_goal=" << dist_to_goal << std::endl;
        }
        
        step++;
        
        // Small delay to make output readable
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    if (step >= max_steps) {
        std::cout << "Simulation completed after " << max_steps << " steps" << std::endl;
    }
    
    std::cout << "Final robot position: " << robot.to_string() << std::endl;
    std::cout << "Trail length: " << robot.trail().size() << " points" << std::endl;
    
    // Test Gaussian Process optimization functionality
    std::cout << "\nTesting Gaussian Process optimization..." << std::endl;
    
    try {
        GPOptimizationParams gp_params;
        gp_params.kernel_variance = config.get_double("optimization", "KERNEL_VARIANCE", 2.0);
        gp_params.kernel_lengthscale = config.get_double("optimization", "KERNEL_LENGTHSCALE", 5.0);
        gp_params.beta = config.get_double("optimization", "BETA", 2.0);
        gp_params.lipschitz = config.get_double("optimization", "LIPSCHITZ", 0.1);
        gp_params.threshold = config.get_double("environment", "THRESHOLD", 0.0);
        gp_params.num_expanders = config.get_int("optimization", "NUM_EXPANDERS", 20);
        
        GaussianOptimizer optimizer(gp_params);
        
        // Set parameter space bounds
        optimizer.set_parameter_bounds(0, 400, 0, 600, 50);
        
        // Initialize with some training data (robot's trail)
        std::vector<Point2D> initial_points;
        std::vector<double> initial_values;
        
        const auto& trail = robot.trail();
        for (size_t i = 0; i < std::min(trail.size(), size_t(10)); ++i) {
            initial_points.push_back(trail[i]);
            // Simulate some objective function value
            double dist_from_goal = distance(trail[i], goal);
            initial_values.push_back(-dist_from_goal); // Negative distance as we want to minimize distance
        }
        
        if (!initial_points.empty()) {
            optimizer.initialize(initial_points, initial_values);
            
            // Get next sampling point
            Point2D next_sample = optimizer.get_next_sample(goal);
            
            auto prediction = optimizer.predict(next_sample);
            std::cout << "Next recommended sampling point: (" << next_sample.x() << ", " << next_sample.y() << ")" << std::endl;
            std::cout << "Predicted value: " << prediction.first << " ± " << prediction.second << std::endl;
            std::cout << "Number of safe parameters: " << optimizer.get_safe_parameters().size() << std::endl;
        }
        
        std::cout << "Gaussian Process optimization test completed!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Gaussian Process test failed: " << e.what() << std::endl;
    }
    
    // Test disjoint set functionality
    std::cout << "\nTesting disjoint set functionality..." << std::endl;
    std::vector<Point2D> test_points = {
        Point2D(0, 0), Point2D(1, 0), Point2D(2, 0),
        Point2D(10, 10), Point2D(11, 10),
        Point2D(20, 20)
    };
    
    DisjointSet ds = build_disjoint_sets(test_points, 2.0);
    std::cout << "Created disjoint sets for " << test_points.size() << " points with threshold 2.0" << std::endl;
    std::cout << "Number of disjoint sets: " << ds.num_sets() << std::endl;
    
    return 0;
}