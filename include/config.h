#pragma once

#include "common.h"
#include <string>
#include <unordered_map>

namespace srpgt {

class Config {
public:
    Config();
    
    // Load configuration from file
    bool load_from_file(const std::string& filename);
    
    // Get configuration values
    double get_double(const std::string& section, const std::string& key, double default_value = 0.0) const;
    int get_int(const std::string& section, const std::string& key, int default_value = 0) const;
    std::string get_string(const std::string& section, const std::string& key, const std::string& default_value = "") const;
    
    // Set configuration values
    void set_double(const std::string& section, const std::string& key, double value);
    void set_int(const std::string& section, const std::string& key, int value);
    void set_string(const std::string& section, const std::string& key, const std::string& value);
    
    // Print configuration for debugging
    void print() const;

private:
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> config_data_;
    
    std::string make_key(const std::string& section, const std::string& key) const;
};

} // namespace srpgt