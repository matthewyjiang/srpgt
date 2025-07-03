# Safe Reactive Navigation for Granular Terrain Exploration

This project provides a simulation framework for safe and reactive navigation of robots in granular terrain environments. It leverages optimization techniques and reactive planning to navigate complex terrains while avoiding obstacles.

**This repository has been ported to C++** for improved performance and easier integration into robotics systems.

## Features

- **Granular Terrain Simulation**: Simulates environments with granular terrain for testing navigation algorithms.
- **Reactive Navigation**: Implements reactive planning for real-time obstacle avoidance.
- **Polygon Geometry Utilities**: Core geometric operations for obstacle representation and manipulation.
- **Disjoint Set Operations**: Efficient connectivity analysis for terrain exploration.
- **Configuration Management**: Flexible configuration system for simulation parameters.

## Prerequisites

### C++ Version (Recommended)
- C++17 compatible compiler (GCC 7+ or Clang 5+)
- CMake 3.12 or newer

### Python Version (Legacy)
- Python 3.9
- Conda (for virtual environment management)

## Setup Instructions

### C++ Build

1. Clone the repository:
   ```bash
   git clone https://github.com/matthewyjiang/srpgt.git
   cd srpgt
   ```

2. Build the project:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

3. Run the simulation:
   ```bash
   ./srpgt ../config.toml
   ```

4. Run tests:
   ```bash
   ./srpgt_test
   ```

### Python Build (Legacy)

1. Create and activate a Conda virtual environment:
   ```bash
   conda create -n reactivenav python=3.9
   conda activate reactivenav
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Simulation

### C++ Version

1. Start the simulation:
   ```bash
   ./srpgt [config_file]
   ```

2. The simulation runs automatically with the following features:
   - Console output showing robot progress
   - Reactive obstacle avoidance
   - Configurable parameters via config file

### Python Version (Legacy)

1. Start the simulation:
   ```bash
   python3 main.py
   ```

2. Use the following controls during the simulation:
   - **Space Bar**: Start/stop the simulation.
   - **Mouse Click**: Set the goal location.
   - **Tab**: Toggle the robot's trail visualization.
   - **F**: Clear captured frames.

## C++ Architecture

The C++ port provides a clean, modular architecture:

- **Robot Class**: Core robot functionality with position tracking and movement
- **Polygon Geometry**: Essential geometric operations for obstacle handling
- **Reactive Planner**: Diffeomorphism-based reactive navigation algorithms
- **Disjoint Sets**: Efficient connectivity analysis for terrain exploration
- **Configuration**: Flexible parameter management

### Key Components

- `include/` - Header files for all components
- `src/` - Implementation files
- `CMakeLists.txt` - Build configuration
- `config.toml` - Simulation parameters

## Visualization

### C++ Version
The C++ version provides console-based output for robot position, goal progress, and navigation status.

### Python Version (Legacy)
The simulation provides a graphical interface where:
- The robot's position and trail are displayed.
- Obstacles and the environment are visualized.
- The goal location can be interactively set.

A sample collage of simulation frames is generated as `collage.png` in the project directory.
