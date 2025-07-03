#pragma once

#include "common.h"
#include <vector>

namespace srpgt {

class Robot {
public:
    Robot(double x, double y, double radius, 
          double screen_width = 800.0, double screen_height = 600.0);
    
    // Update robot position with velocity
    void update(const Vector2D& velocity);
    
    // Clear the trail
    void clear_trail();
    
    // Enable/disable trail tracking
    void set_trail_enable(bool enable) { trail_enable_ = enable; }
    bool is_trail_enabled() const { return trail_enable_; }
    
    // Getters
    const Point2D& position() const { return pos_; }
    double radius() const { return radius_; }
    double angle() const { return angle_; }
    const std::vector<Point2D>& trail() const { return trail_; }
    
    // Setters
    void set_position(const Point2D& pos) { pos_ = pos; }
    void set_move_speed(double speed) { move_speed_ = speed; }
    void set_max_move_speed(double max_speed) { max_move_speed_ = max_speed; }
    
    // Get robot state as string for debugging
    std::string to_string() const;

private:
    std::vector<Point2D> trail_;
    bool trail_enable_;
    Point2D pos_;
    double radius_;
    double angle_;
    double move_speed_;
    double max_move_speed_;
    double rotation_speed_;
    double screen_width_;
    double screen_height_;
};

} // namespace srpgt