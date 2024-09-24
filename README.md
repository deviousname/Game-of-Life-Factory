"""
This is currently an incremental game which the uses can generate automated energy farms with. I have plans to expand
this into a hybrid game, where the factory is a building on the overworld, the player can move a character with wasd,
enter and exit the factory to tweak it as needed, and use their energy for things, and use their colors for stats,
for example, having a lot of red will make you strong against fire, a lot of blue, strong against water, etc.
More energy means more ability use. Other factories and characters will exist in the overworld to battle and learn
factory secrets from.

This module defines a comprehensive simulation framework for cellular automata within the context of factory design,
leveraging Pygame for graphical representation and Numba for performance optimization. The system supports multiple 
automata rulesets, dynamic color manipulation, and player interaction, offering a rich environment to visualize and 
manipulate game states in real-time.

### Key Features:
1. **Serialization & Deserialization**: 
   - Seamlessly encode and decode simulation states using compression and base64 encoding, allowing for easy sharing 
     of game seeds or restoration of previous states.
   
2. **Grid-Based Automata**: 
   - Implements cellular automata with various rulesets like Conway’s Game of Life, HighLife, and more, providing both 
     building and simulation modes for diverse gameplay.
   
3. **Dynamic Color Handling**: 
   - Generates and assigns colors to cells dynamically, while also handling color-based interactions, like tallying 
     living cells based on their proximity to primary colors.

4. **Inventory & Ruleset Management**: 
   - Enables player interaction with automata rulesets, including block-based inventory management, and cycling through 
     available rulesets for gameplay variation.

5. **Interactive UI**: 
   - Includes a full-featured graphical interface for toggling between modes, purchasing logic blocks, handling grid 
     interactions, and pausing or resetting simulations.

6. **Clipboard Support**: 
   - Provides functionality for copying and pasting simulation states using the clipboard, making sharing and restoring 
     states intuitive.

### Core Classes and Functions:
- **Factory_View**: 
   - The main class managing grid display, event handling, and simulation logic.
   
- **TextInputBox**: 
   - A utility class for text input, used for saving and loading game states via seed strings.

- **Serialization/Deserialization**: 
   - Helper functions `serialize_state` and `deserialize_state` enable conversion of the game state to/from a 
     compressed, shareable string format.

### Numba Optimizations:
- Several functions are accelerated using Numba's `njit` decorator to improve performance, especially in grid manipulations 
  and neighbor calculations, critical for the cellular automata’s efficiency.

"""
