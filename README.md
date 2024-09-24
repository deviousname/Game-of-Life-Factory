"""
This script implements a grid-based simulation tool using Pygame to visualize and interact with cellular automata systems, such as Conway's Game of Life. It provides a graphical interface for building custom simulation grids with various rulesets and allows users to dynamically switch between a building mode and a simulation mode. The primary functionality includes:

1. **Grid Customization**: Users can design and configure the grid layout, defining areas of the grid with different rulesets. Each ruleset determines how cells live, die, or replicate based on neighboring cell states.

2. **Rulesets**: The script includes several predefined rulesets, such as Conway's Game of Life, HighLife, and others. Users can switch between these rulesets to experiment with different behaviors in the simulation.

3. **Color System**: The cells are visually represented using a color system where each cell state is mapped to a color. The system dynamically adjusts cell colors based on simulation outcomes, and users can change the color scheme while in simulation mode.

4. **Clipboard Integration**: The application supports saving and loading grid states using serialized data, which can be copied to and pasted from the system clipboard.

5. **Interactive Simulation**: Users can pause, reset, and modify the simulation in real-time, allowing for hands-on experimentation with different grid configurations and rulesets.

6. **Serialization & Deserialization**: Grid states (logic grid and cell states) can be serialized into a compressed base64 string, enabling easy sharing or persistence of grid configurations.

7. **Optimization**: The script uses Numba for performance optimization during the simulation update process, handling large grids efficiently by leveraging just-in-time (JIT) compilation.

The application allows users to explore various cellular automata models with an intuitive visual interface while providing mechanisms to save, share, and reload grid states.
"""
