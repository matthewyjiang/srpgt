#include "robot.h"
#include <sstream>
#include <cmath>

namespace srpgt {

Robot::Robot(double x, double y, double radius, 
             double screen_width, double screen_height)
    : trail_enable_(true)
    , pos_(x, y)
    , radius_(radius)
    , angle_(0.0)
    , move_speed_(0.2)
    , max_move_speed_(2.5)
    , rotation_speed_(0.04)
    , screen_width_(screen_width)
    , screen_height_(screen_height) {
}

void Robot::update(const Vector2D& velocity) {
    // Add current position to trail
    if (trail_enable_) {
        trail_.push_back(pos_);
    }
    
    // Calculate direction angle
    angle_ = std::atan2(velocity.y(), velocity.x());
    
    // Update position with speed limiting
    Vector2D term = velocity * move_speed_;
    if (term.norm() > max_move_speed_) {
        term = term.normalized() * max_move_speed_;
    }
    
    pos_ += term;
    
    // Keep robot within screen boundaries (commented out like in original)
    // pos_.x() = std::max(radius_, std::min(screen_width_ - radius_, pos_.x()));
    // pos_.y() = std::max(radius_, std::min(screen_height_ - radius_, pos_.y()));
}

void Robot::clear_trail() {
    trail_.clear();
}

std::string Robot::to_string() const {
    std::ostringstream oss;
    oss << "Robot[pos=(" << pos_.x() << "," << pos_.y() 
        << "), angle=" << angle_ 
        << ", radius=" << radius_ 
        << ", trail_points=" << trail_.size() << "]";
    return oss.str();
}

} // namespace srpgt