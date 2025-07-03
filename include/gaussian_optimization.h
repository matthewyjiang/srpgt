#pragma once

#include "common.h"
#include <gaussian_process.h>
#include <vector>

namespace srpgt {

// Configuration parameters for Gaussian process optimization
struct GPOptimizationParams {
    double kernel_variance = 2.0;
    double kernel_lengthscale = 5.0;
    double beta = 2.0;
    double lipschitz = 0.1;
    double threshold = 0.0;
    int num_expanders = 20;
};

// SafeOpt-like functionality using Gaussian Process
class GaussianOptimizer {
public:
    GaussianOptimizer(const GPOptimizationParams& params);
    
    // Initialize with training data
    void initialize(const std::vector<Point2D>& initial_points, 
                   const std::vector<double>& initial_values);
    
    // Add new observation
    void add_observation(const Point2D& point, double value);
    
    // Get next sampling point (SafeOpt acquisition function)
    Point2D get_next_sample(const Point2D& goal, double rho = 1.0);
    
    // Predict at a given point
    std::pair<double, double> predict(const Point2D& point) const;
    
    // Get safe parameter set
    std::vector<Point2D> get_safe_parameters() const;
    
    // Set parameter space bounds
    void set_parameter_bounds(double min_x, double max_x, double min_y, double max_y, int resolution = 100);

private:
    GPOptimizationParams params_;
    std::unique_ptr<gp::GaussianProcess> gp_;
    std::vector<Point2D> parameter_set_;
    std::vector<Point2D> safe_set_;
    bool initialized_;
    
    // Update confidence intervals and safe set
    void update_confidence_intervals();
    
    // Compute acquisition function value
    double acquisition_function(const Point2D& point, const Point2D& goal, double rho) const;
};

} // namespace srpgt