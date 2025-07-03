#include "config.h"
#include <fstream>
#include <sstream>
#include <iostream>

namespace srpgt {

Config::Config() {
    // Set default values
    set_string("environment", "FILENAME", "data/terrain.csv");
    set_int("environment", "THRESHOLD", 500);
    set_int("environment", "SIMPLIFICATION_CONSTANT", 4);
    set_string("environment", "MODE", "semnav");
    
    set_double("robot", "ROBOT_RADIUS", 2.0);
    set_string("robot", "MODE", "navigate");
    set_double("robot", "STARTING_X", 40.0);
    set_double("robot", "STARTING_Y", 221.0);
    set_double("robot", "GOAL_X", 185.0);
    set_double("robot", "GOAL_Y", 360.0);
    
    set_int("optimization", "NUM_EXPANDERS", 120);
    set_double("optimization", "KERNEL_VARIANCE", 2.0);
    set_double("optimization", "KERNEL_LENGTHSCALE", 30.0);
    set_double("optimization", "BETA", 3.0);
    set_double("optimization", "LIPSCHITZ", 0.003);
    set_double("optimization", "RHO", 0.0);
    
    set_int("display", "BUFFER_SIZE", 1);
    set_int("display", "FRAMECOUNT", 1);
    set_int("display", "COLUMNS", 1);
}

bool Config::load_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open config file: " << filename << std::endl;
        return false;
    }
    
    std::string line, current_section;
    while (std::getline(file, line)) {
        // Remove whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Check for section header
        if (line[0] == '[' && line.back() == ']') {
            current_section = line.substr(1, line.length() - 2);
            continue;
        }
        
        // Parse key-value pair
        size_t eq_pos = line.find('=');
        if (eq_pos != std::string::npos && !current_section.empty()) {
            std::string key = line.substr(0, eq_pos);
            std::string value = line.substr(eq_pos + 1);
            
            // Remove whitespace from key and value
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            // Remove quotes from value if present
            if (value.length() >= 2 && value[0] == '"' && value.back() == '"') {
                value = value.substr(1, value.length() - 2);
            }
            
            config_data_[current_section][key] = value;
        }
    }
    
    return true;
}

double Config::get_double(const std::string& section, const std::string& key, double default_value) const {
    auto section_it = config_data_.find(section);
    if (section_it == config_data_.end()) return default_value;
    
    auto key_it = section_it->second.find(key);
    if (key_it == section_it->second.end()) return default_value;
    
    try {
        return std::stod(key_it->second);
    } catch (const std::exception&) {
        return default_value;
    }
}

int Config::get_int(const std::string& section, const std::string& key, int default_value) const {
    auto section_it = config_data_.find(section);
    if (section_it == config_data_.end()) return default_value;
    
    auto key_it = section_it->second.find(key);
    if (key_it == section_it->second.end()) return default_value;
    
    try {
        return std::stoi(key_it->second);
    } catch (const std::exception&) {
        return default_value;
    }
}

std::string Config::get_string(const std::string& section, const std::string& key, const std::string& default_value) const {
    auto section_it = config_data_.find(section);
    if (section_it == config_data_.end()) return default_value;
    
    auto key_it = section_it->second.find(key);
    if (key_it == section_it->second.end()) return default_value;
    
    return key_it->second;
}

void Config::set_double(const std::string& section, const std::string& key, double value) {
    config_data_[section][key] = std::to_string(value);
}

void Config::set_int(const std::string& section, const std::string& key, int value) {
    config_data_[section][key] = std::to_string(value);
}

void Config::set_string(const std::string& section, const std::string& key, const std::string& value) {
    config_data_[section][key] = value;
}

void Config::print() const {
    for (const auto& section : config_data_) {
        std::cout << "[" << section.first << "]" << std::endl;
        for (const auto& kv : section.second) {
            std::cout << kv.first << " = " << kv.second << std::endl;
        }
        std::cout << std::endl;
    }
}

} // namespace srpgt