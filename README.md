# Safe Reactive Navigation for Granular Terrain Exploration

This project provides a simulation framework for safe and reactive navigation of robots in granular terrain environments. It leverages optimization techniques and reactive planning to navigate complex terrains while avoiding obstacles.

## Features

- **Granular Terrain Simulation**: Simulates environments with granular terrain for testing navigation algorithms.
- **Reactive Navigation**: Implements reactive planning for real-time obstacle avoidance.
- **Visualization**: Provides a graphical interface for visualizing the robot's navigation and environment.
- **Goal Setting**: Allows users to set navigation goals interactively.
- **Collage Creation**: Captures simulation frames and generates a collage for analysis.

## Prerequisites

- Python 3.12
- uv (for Python package and environment management)

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone git@github.com:matthewyjiang/srpgt.git
   ```

2. Install dependencies with uv:
   ```bash
   cd srpgt
   uv sync
   ```


## Running the Simulation

1. Start the simulation:
   ```bash
   uv run reactive-navigation
   ```

2. Use the following controls during the simulation:
   - **Space Bar**: Start/stop the simulation.
   - **Mouse Click**: Set the goal location.
   - **Tab**: Toggle the robot's trail visualization.
   - **F**: Clear captured frames.


## Visualization

The simulation provides a graphical interface where:
- The robot's position and trail are displayed.
- Obstacles and the environment are visualized.
- The goal location can be interactively set.

A sample collage of simulation frames is generated as `collage.png` in the project directory.
