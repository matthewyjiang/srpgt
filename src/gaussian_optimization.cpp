#include "gaussian_optimization.h"
#include <rbf_kernel.h>
#include <cmath>
#include <algorithm>
#include <random>

namespace srpgt {

GaussianOptimizer::GaussianOptimizer(const GPOptimizationParams& params)
    : params_(params), initialized_(false) {
    // Create Gaussian Process with RBF kernel
    auto kernel = std::make_unique<gp::RBFKernel>(params_.kernel_variance, params_.kernel_lengthscale);
    gp_ = std::make_unique<gp::GaussianProcess>(std::move(kernel));
}

void GaussianOptimizer::initialize(const std::vector<Point2D>& initial_points, 
                                  const std::vector<double>& initial_values) {
    if (initial_points.size() != initial_values.size()) {
        throw std::invalid_argument("Initial points and values must have same size");
    }
    
    // Convert Point2D to Eigen format for GP library
    Eigen::MatrixXd X_train(initial_points.size(), 2);
    Eigen::VectorXd y_train(initial_values.size());
    
    for (size_t i = 0; i < initial_points.size(); ++i) {
        X_train(i, 0) = initial_points[i].x();
        X_train(i, 1) = initial_points[i].y();
        y_train(i) = initial_values[i];
    }
    
    // Fit the Gaussian Process
    gp_->fit(X_train, y_train);
    
    initialized_ = true;
    update_confidence_intervals();
}

void GaussianOptimizer::add_observation(const Point2D& point, double value) {
    if (!initialized_) {
        throw std::runtime_error("Optimizer not initialized");
    }
    
    Eigen::VectorXd x(2);
    x << point.x(), point.y();
    gp_->add_data_point(x, value);
    
    update_confidence_intervals();
}

Point2D GaussianOptimizer::get_next_sample(const Point2D& goal, double rho) {
    if (!initialized_) {
        throw std::runtime_error("Optimizer not initialized");
    }
    
    Point2D best_point = parameter_set_[0];
    double best_acquisition = -std::numeric_limits<double>::infinity();
    
    // Evaluate acquisition function over parameter set
    for (const auto& point : safe_set_) {
        double acq_value = acquisition_function(point, goal, rho);
        if (acq_value > best_acquisition) {
            best_acquisition = acq_value;
            best_point = point;
        }
    }
    
    return best_point;
}

std::pair<double, double> GaussianOptimizer::predict(const Point2D& point) const {
    if (!initialized_) {
        throw std::runtime_error("Optimizer not initialized");
    }
    
    Eigen::MatrixXd x(1, 2);
    x(0, 0) = point.x();
    x(0, 1) = point.y();
    
    auto prediction = gp_->predict(x, true);
    return {prediction.first(0), std::sqrt(prediction.second(0))};
}

std::vector<Point2D> GaussianOptimizer::get_safe_parameters() const {
    return safe_set_;
}

void GaussianOptimizer::set_parameter_bounds(double min_x, double max_x, double min_y, double max_y, int resolution) {
    parameter_set_.clear();
    
    double dx = (max_x - min_x) / resolution;
    double dy = (max_y - min_y) / resolution;
    
    for (int i = 0; i <= resolution; ++i) {
        for (int j = 0; j <= resolution; ++j) {
            double x = min_x + i * dx;
            double y = min_y + j * dy;
            parameter_set_.emplace_back(x, y);
        }
    }
}

void GaussianOptimizer::update_confidence_intervals() {
    safe_set_.clear();
    
    for (const auto& point : parameter_set_) {
        auto pred = predict(point);
        double mean = pred.first;
        double std_dev = pred.second;
        
        // Lower confidence bound (conservative estimate)
        double lower_bound = mean - params_.beta * std_dev;
        
        if (lower_bound >= params_.threshold) {
            safe_set_.push_back(point);
        }
    }
}

double GaussianOptimizer::acquisition_function(const Point2D& point, const Point2D& goal, double rho) const {
    auto pred = predict(point);
    double mean = pred.first;
    double std_dev = pred.second;
    
    // Upper confidence bound for exploration
    double ucb = mean + params_.beta * std_dev;
    
    // Distance to goal (for exploitation)
    double dist_to_goal = distance(point, goal);
    double goal_term = -rho * dist_to_goal;
    
    return ucb + goal_term;
}

} // namespace srpgt