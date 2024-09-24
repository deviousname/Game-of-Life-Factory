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


import pygame
import numpy as np
import colorsys
from numba import njit
import base64
import zlib
import pyperclip
import hashlib

# Serialization functions
def serialize_state(logic_grid, cell_state_grid):
    # Convert grids to bytes
    logic_bytes = logic_grid.tobytes()
    cell_state_bytes = cell_state_grid.tobytes()

    # Concatenate bytes
    combined_bytes = logic_bytes + cell_state_bytes

    # Compress the bytes
    compressed_bytes = zlib.compress(combined_bytes)

    # Encode as base64 to get a string
    key = base64.b64encode(compressed_bytes).decode('utf-8')
    return key

def deserialize_state(key, grid_shape):
    # Decode from base64
    compressed_bytes = base64.b64decode(key.encode('utf-8'))

    # Decompress the bytes
    combined_bytes = zlib.decompress(compressed_bytes)

    # Calculate the split index
    grid_size = grid_shape[0] * grid_shape[1] * np.dtype(np.int32).itemsize
    logic_bytes = combined_bytes[:grid_size]
    cell_state_bytes = combined_bytes[grid_size:]

    # Convert bytes back to NumPy arrays, ensure they are writable
    logic_grid = np.frombuffer(logic_bytes, dtype=np.int32).reshape(grid_shape).copy()  # Make writable
    cell_state_grid = np.frombuffer(cell_state_bytes, dtype=np.int32).reshape(grid_shape).copy()  # Make writable

    return logic_grid, cell_state_grid

# Define available rulesets
RULESETS = {
    "Void": {"B": [], "S": []},  # All cells die
    "Conway": {"B": [3], "S": [2, 3]},  # Standard Game of Life
    "HighLife": {"B": [3, 6], "S": [2, 3]},  # HighLife
    "DayAndNight": {"B": [3, 6, 7, 8], "S": [3, 4, 6, 7, 8]},  # Day & Night
    "Seeds": {"B": [2], "S": []},  # Seeds
    "LifeWithoutDeath": {"B": [3], "S": list(range(9))},  # Life without Death
    "Maze": {"B": [3], "S": [1, 2, 3, 4, 5]},  # Maze
    "Gnarl": {"B": [1], "S": [1]},  # Gnarl
    "Replicator": {"B": [1, 3, 5, 7], "S": [1, 3, 5, 7]},  # Replicator
}

# Map ruleset names to integer IDs
RULESET_IDS = {name: idx for idx, name in enumerate(RULESETS.keys())}
ID_RULESETS = {idx: name for name, idx in RULESET_IDS.items()}

# Define primary colors in RGB space
PRIMARY_COLORS = [
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (255, 255, 0), # Yellow
    (0, 255, 255), # Cyan
    (255, 0, 255)  # Magenta
]

def color_distance(c1, c2):
    """Calculate Euclidean distance between two RGB colors."""
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

def restructure_colors(colors_list):
    # For each color, find its closest primary color and sort based on distance
    sorted_colors = sorted(
        colors_list, 
        key=lambda color: min(color_distance(color, primary) for primary in PRIMARY_COLORS)
    )
    return sorted_colors

class Factory_View:
    def __init__(
        self,
        grid_size=(50, 100),
        cell_size=16,
        margin=1,
        n_colors=255,
        window_title="Factory View",
        fullscreen=False,
        fps_drawing=240,  # FPS for rendering the screen
        fps_simulation=240  # FPS for simulation updates
    ):
        # Initialize Pygame
        pygame.init()
        self.fullscreen = fullscreen
        if fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            display_width, display_height = self.screen.get_size()
            
            # Calculate the maximum square size that fits within the screen
            max_display_size = min(display_width, display_height)
            
            # Compute the new cell size to maintain the square ratio
            self.cell_size = (max_display_size - margin) // max(grid_size)

            # Update the screen width and height for square grid centered alignment
            self.width = self.height = self.cell_size * max(grid_size) + margin * (max(grid_size) + 1)
            
            # Create a smaller surface to draw the grid and then center it
            self.grid_surface = pygame.Surface((self.width, self.height))
        else:
            self.width = grid_size[1] * (cell_size + margin) + margin
            self.height = grid_size[0] * (cell_size + margin) + margin + 30  # Extra space for UI
            self.screen = pygame.display.set_mode((self.width, self.height))

        pygame.display.set_caption(window_title)
        
        # Grid settings
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.margin = margin
        
        # Simulation and drawing FPS control
        self.fps_drawing = fps_drawing
        self.fps_simulation = fps_simulation
        self.simulation_time_accumulator = 0.0  # To track the simulation time
        self.simulation_interval = 1.0 / self.fps_simulation  # Time between each simulation update
        
        # Generate colors (from proof of concept code)
        self.colors_list = [
            tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, 1, 1))
            for h in np.linspace(0, 1, n_colors, endpoint=False)
        ]
        self.colors_list.append((0, 0, 0))  # Black color for dead cells

        # Restructure the colors list by proximity to primary colors
        restructured_colors_list = restructure_colors(self.colors_list)

        # Convert to numpy array again if needed
        self.colors_array = np.array(restructured_colors_list, dtype=np.uint8)
        self.black_index = len(self.colors_list) - 1  # Index of the black color (unchanged)

        # Generate logic colors for better differentiation
        self.logic_colors_list = [
            tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, 1, 1))
            for h in np.linspace(0, 1, len(RULESETS), endpoint=False)
        ]

        # Color selection for simulation mode
        self.selected_color_index = 0  # Start with the first color

        # Ruleset selection
        self.selected_ruleset_name = "Conway"
        self.selected_ruleset_id = RULESET_IDS[self.selected_ruleset_name]

        # Initialize grid data structures
        self.initialize_grids()

        # Preprocess rulesets for Numba
        self.preprocess_rulesets()

        # Modes
        self.mode = 'building'  # Modes: 'building', 'simulation', 'menu'
        self.paused = True
        
        # Other settings
        self.clock = pygame.time.Clock()
        self.done = False

        # Neighbor offsets for simulation
        self.neighbor_offsets = np.array([
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ], dtype=np.int32)

        # Mouse button states
        self.mouse_buttons = [False, False, False]  # Left, Middle, Right buttons

        # Initialize clipboard support
        try:
            import pyperclip
            self.clipboard_available = True
        except ImportError:
            self.clipboard_available = False
            print("pyperclip module not found. Copy to clipboard will not work.")


    def preprocess_rulesets(self):
        """Preprocess rulesets into Numba-compatible arrays."""
        num_rulesets = len(RULESETS)
        max_rule_length = max(len(ruleset["B"]) + len(ruleset["S"]) for ruleset in RULESETS.values())
        self.birth_rules_array = np.full((num_rulesets, max_rule_length), -1, dtype=np.int32)
        self.survival_rules_array = np.full((num_rulesets, max_rule_length), -1, dtype=np.int32)
        self.rule_lengths = np.zeros((num_rulesets, 2), dtype=np.int32)  # Stores lengths of B and S rules

        for idx, name in ID_RULESETS.items():
            ruleset = RULESETS[name]
            B = np.array(ruleset["B"], dtype=np.int32)
            S = np.array(ruleset["S"], dtype=np.int32)
            self.birth_rules_array[idx, :len(B)] = B
            self.survival_rules_array[idx, :len(S)] = S
            self.rule_lengths[idx] = [len(B), len(S)]

    def run(self):
        """Main loop of the application."""
        while not self.done:
            self.handle_events()
            
            # Track the time since the last simulation step
            dt = self.clock.tick(self.fps_drawing) / 1000.0  # Time passed since the last frame
            self.simulation_time_accumulator += dt

            # Update the simulation if enough time has passed
            if not self.paused and self.simulation_time_accumulator >= self.simulation_interval:
                self.update()
                self.simulation_time_accumulator -= self.simulation_interval

            # Draw the current state
            self.draw()
            pygame.display.update()

    def handle_events(self):
        """Handle user input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
                pygame.quit()
                exit()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if self.mode != 'menu':
                    self.previous_mode = self.mode
                    self.mode = 'menu'
                    self.paused = True
                else:
                    self.mode = self.previous_mode

            if self.mode == 'menu':
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        mouse_pos = pygame.mouse.get_pos()
                        # Copy button
                        if self.copy_button_rect.collidepoint(mouse_pos):
                            key = serialize_state(self.logic_grid, self.cell_state_grid)
                            if self.clipboard_available:
                                pyperclip.copy(key)
                                print("Seed copied to clipboard.")
                            else:
                                print("Clipboard not available.")
                        # Paste button
                        if self.paste_button_rect.collidepoint(mouse_pos):
                            if self.clipboard_available:
                                try:
                                    clipboard_content = pyperclip.paste()
                                    try:
                                        self.logic_grid, self.cell_state_grid = deserialize_state(clipboard_content, self.grid_size)
                                        print("Seed loaded from clipboard.")
                                    except Exception as e:
                                        print(f"Error loading seed: {e}. Using clipboard content as seed.")
                                        # Use the clipboard content as a deterministic seed
                                        self.generate_state_from_seed(clipboard_content)
                                except Exception as e:
                                    print(f"Error accessing clipboard: {e}")
                            else:
                                print("Clipboard not available.")
            else:
                # Handle events in other modes
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_TAB:
                        self.mode = 'simulation' if self.mode == 'building' else 'building'
                    elif event.key == pygame.K_SPACE and self.mode == 'simulation':
                        self.paused = not self.paused
                    elif event.key == pygame.K_d:
                        if self.mode == 'building':
                            ruleset_names = list(RULESETS.keys())
                            idx = ruleset_names.index(self.selected_ruleset_name)
                            idx = (idx + 1) % len(ruleset_names)
                            self.selected_ruleset_name = ruleset_names[idx]
                            self.selected_ruleset_id = RULESET_IDS[self.selected_ruleset_name]
                        elif self.mode == 'simulation':
                            self.selected_color_index = (self.selected_color_index + 1) % (len(self.colors_list) - 1)
                    elif event.key == pygame.K_a:
                        if self.mode == 'building':
                            ruleset_names = list(RULESETS.keys())
                            idx = ruleset_names.index(self.selected_ruleset_name)
                            idx = (idx - 1) % len(ruleset_names)
                            self.selected_ruleset_name = ruleset_names[idx]
                            self.selected_ruleset_id = RULESET_IDS[self.selected_ruleset_name]
                        elif self.mode == 'simulation':
                            self.selected_color_index = (self.selected_color_index - 1) % (len(self.colors_list) - 1)
                    elif event.key == pygame.K_f:
                        self.handle_flood_fill()
                    elif event.key == pygame.K_r:
                        self.reset_game()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button <= 3:
                        self.mouse_buttons[event.button - 1] = True
                    self.handle_mouse_event()
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button <= 3:
                        self.mouse_buttons[event.button - 1] = False
                elif event.type == pygame.MOUSEMOTION:
                    if any(self.mouse_buttons):
                        self.handle_mouse_event()

    def reset_game(self):
        """Reset the game grid based on the current mode."""
        if self.mode == 'building':
            # Only reset the logic grid for the building mode
            self.initialize_logic_grid()
            self.selected_ruleset_name = "Conway"  # Reset the selected ruleset to the default
            self.selected_ruleset_id = RULESET_IDS[self.selected_ruleset_name]
        elif self.mode == 'simulation':
            # Only reset the cell state grid for the simulation mode
            self.initialize_cell_state_grid()
            self.paused = True  # Optionally pause the simulation when resetting

    def handle_flood_fill(self):
        """Handle flood fill action based on mode."""
        pos = pygame.mouse.get_pos()
        col = pos[0] // (self.cell_size + self.margin)
        row = (pos[1] - 30) // (self.cell_size + self.margin)  # Adjust for UI height

        if 0 <= row < self.grid_size[0] and 0 <= col < self.grid_size[1]:
            if self.mode == 'building':
                target_logic = self.logic_grid[row, col]
                if target_logic != -1:  # Only flood fill cells with assigned logic
                    self.flood_fill_logic(row, col, target_logic)
            elif self.mode == 'simulation':
                target_color = self.cell_state_grid[row, col]
                self.flood_fill_color(row, col, target_color)

    def initialize_grids(self):
        """Initialize both logic and cell state grids."""
        self.initialize_logic_grid()
        self.initialize_cell_state_grid()

    def initialize_logic_grid(self):
        """Initialize only the logic grid."""
        rows, cols = self.grid_size
        self.logic_grid = np.full((rows, cols), fill_value=0, dtype=np.int32).copy()  # Ensure a writable array

    def initialize_cell_state_grid(self):
        """Initialize only the cell state grid."""
        rows, cols = self.grid_size
        self.cell_state_grid = np.full((rows, cols), self.black_index, dtype=np.int32).copy()  # Ensure writable

    def flood_fill_logic(self, row, col, target_logic):
        """Flood fill logic for adjacent cells, including logicless cells (Void, 0)."""
        
        # If the selected cell is already the same ruleset, return
        if self.logic_grid[row, col] == self.selected_ruleset_id:
            return

        # The stack holds cells to be filled
        stack = [(row, col)]
        
        # This is the value of the block where the fill starts (can be Void or a valid ruleset)
        initial_value = self.logic_grid[row, col]

        while stack:
            r, c = stack.pop()
            if 0 <= r < self.grid_size[0] and 0 <= c < self.grid_size[1]:
                # Allow filling either cells with the initial value or cells that are Void (0)
                if self.logic_grid[r, c] == initial_value or self.logic_grid[r, c] == 0:
                    self.logic_grid[r, c] = self.selected_ruleset_id  # Apply selected ruleset
                    neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
                    for nr, nc in neighbors:
                        if 0 <= nr < self.grid_size[0] and 0 <= nc < self.grid_size[1]:
                            if self.logic_grid[nr, nc] == initial_value or self.logic_grid[nr, nc] == 0:
                                stack.append((nr, nc))
                      
    def flood_fill_color(self, row, col, target_color):
        """Flood fill color for adjacent cells."""
        if self.cell_state_grid[row, col] == self.selected_color_index:
            # Do nothing if the selected cell is already the target color
            return

        stack = [(row, col)]
        initial_color = self.cell_state_grid[row, col]

        while stack:
            r, c = stack.pop()
            if 0 <= r < self.grid_size[0] and 0 <= c < self.grid_size[1]:
                if self.cell_state_grid[r, c] == initial_color:
                    self.cell_state_grid[r, c] = self.selected_color_index
                    neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
                    for nr, nc in neighbors:
                        if 0 <= nr < self.grid_size[0] and 0 <= nc < self.grid_size[1]:
                            if self.cell_state_grid[nr, nc] == initial_color:
                                stack.append((nr, nc))

    def handle_mouse_event(self):
        """Handle mouse interactions."""
        pos = pygame.mouse.get_pos()
        col = pos[0] // (self.cell_size + self.margin)
        row = (pos[1] - 30) // (self.cell_size + self.margin)  # Adjust for UI height
        if 0 <= row < self.grid_size[0] and 0 <= col < self.grid_size[1]:
            if self.mode == 'building':
                if self.mouse_buttons[0]:  # Left click to assign logic
                    self.logic_grid[row, col] = self.selected_ruleset_id
                elif self.mouse_buttons[2]:  # Right click to erase logic (set to Void state)
                    self.logic_grid[row, col] = 0  # Set back to Void logic
            elif self.mode == 'simulation':
                if self.mouse_buttons[0]:  # Left click to paint living cells
                    self.cell_state_grid[row, col] = self.selected_color_index
                elif self.mouse_buttons[2]:  # Right click to erase cells
                    self.cell_state_grid[row, col] = self.black_index

    def update(self):
        """Update the simulation."""
        if self.mode == 'simulation' and not self.paused:
            self.cell_state_grid = update_cells(
                self.cell_state_grid,
                self.logic_grid,
                self.birth_rules_array,
                self.survival_rules_array,
                self.rule_lengths,
                self.black_index,
                self.neighbor_offsets,
                self.colors_array
            )
            
    def draw_menu(self):
        # Draw semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # Semi-transparent black
        self.screen.blit(overlay, (0, 0))

        # Draw the popup box
        box_width = 600
        box_height = 200
        box_x = (self.width - box_width) // 2
        box_y = (self.height - box_height) // 2
        popup_rect = pygame.Rect(box_x, box_y, box_width, box_height)
        pygame.draw.rect(self.screen, (50, 50, 50), popup_rect)

        # Copy button centered
        font = pygame.font.Font(None, 24)
        copy_button = pygame.Rect(box_x + box_width // 4 - 30, box_y + 60, 60, 30)
        pygame.draw.rect(self.screen, (70, 70, 70), copy_button)
        copy_text = font.render('Copy', True, (255, 255, 255))
        copy_text_rect = copy_text.get_rect(center=copy_button.center)
        self.screen.blit(copy_text, copy_text_rect)

        # Paste button centered
        paste_button = pygame.Rect(box_x + 3 * box_width // 4 - 30, box_y + 60, 60, 30)
        pygame.draw.rect(self.screen, (70, 70, 70), paste_button)
        paste_text = font.render('Paste', True, (255, 255, 255))
        paste_text_rect = paste_text.get_rect(center=paste_button.center)
        self.screen.blit(paste_text, paste_text_rect)

        # Store the copy and paste button rects for click detection
        self.copy_button_rect = copy_button
        self.paste_button_rect = paste_button

        # Instruction
        instruction_text = font.render('Use Copy to copy the seed and Paste to load it.', True, (255, 255, 255))
        instruction_rect = instruction_text.get_rect(center=(self.width // 2, box_y + 120))
        self.screen.blit(instruction_text, instruction_rect)
        
    def draw(self):
        """Draw the grid and UI elements."""
        self.screen.fill((0, 0, 0))  # Clear screen

        if self.mode == 'menu':
            # Draw the game in the background
            self.draw_game()
            self.draw_menu()
        else:
            self.draw_game()

    def draw_game(self):
        if self.fullscreen:
            # Draw to the smaller centered surface
            self.grid_surface.fill((0, 0, 0))

            # Draw grid
            for row in range(self.grid_size[0]):
                for col in range(self.grid_size[1]):
                    x = col * (self.cell_size + self.margin) + self.margin
                    y = row * (self.cell_size + self.margin) + self.margin
                    if self.mode == 'building':
                        ruleset_id = self.logic_grid[row, col]
                        if ruleset_id == 0:  # Handle Void (empty logic)
                            color = (50, 50, 50)  # Dark gray for Void state
                        else:
                            color = self.logic_colors_list[ruleset_id]
                    else:
                        color_index = self.cell_state_grid[row, col]
                        color = self.colors_array[color_index]
                    pygame.draw.rect(
                        self.grid_surface,
                        color,
                        (x, y, self.cell_size, self.cell_size)
                    )

            # Blit the centered grid surface onto the screen
            grid_x = (self.screen.get_width() - self.grid_surface.get_width()) // 2
            grid_y = (self.screen.get_height() - self.grid_surface.get_height()) // 2
            self.screen.blit(self.grid_surface, (grid_x, grid_y))
        else:
            # Draw grid directly on the screen
            for row in range(self.grid_size[0]):
                for col in range(self.grid_size[1]):
                    x = col * (self.cell_size + self.margin) + self.margin
                    y = row * (self.cell_size + self.margin) + self.margin + 30  # Adjust for UI height
                    if self.mode == 'building':
                        ruleset_id = self.logic_grid[row, col]
                        if ruleset_id == 0:  # Handle Void (empty logic)
                            color = (50, 50, 50)  # Dark gray for Void state
                        else:
                            color = self.logic_colors_list[ruleset_id]
                    else:
                        color_index = self.cell_state_grid[row, col]
                        color = self.colors_array[color_index]
                    pygame.draw.rect(
                        self.screen,
                        color,
                        (x, y, self.cell_size, self.cell_size)
                    )

        # Draw UI elements (adjust positioning as needed)
        font = pygame.font.SysFont(None, 24)
        mode_text = font.render(f"Mode: {self.mode.capitalize()}", True, (255, 255, 255))
        self.screen.blit(mode_text, (10, 5))

        if self.mode == 'building':
            ruleset_text = font.render(f"Selected Ruleset: {self.selected_ruleset_name}", True, (255, 255, 255))
            self.screen.blit(ruleset_text, (200, 5))
            instructions = font.render("Tab: Switch Mode | A and D: Change Ruleset | R: Reset", True, (255, 255, 255))
        else:
            selected_color = self.colors_list[self.selected_color_index]
            color_text = font.render(f"Selected Color: {selected_color}", True, (255, 255, 255))
            self.screen.blit(color_text, (200, 5))
            instructions = font.render("Tab: Switch Mode | A and D: Change Color | R: Reset  | Space: Pause", True, (255, 255, 255))

        self.screen.blit(instructions, (10, self.height - 25))
            
    def handle_save(self):
        # Generate the key
        key = serialize_state(self.logic_grid, self.cell_state_grid)

        # Create a text input box with the key
        self.save_textbox = TextInputBox(100, 300, 400, 32, text=key)

    def handle_load(self):
        # Create an empty text input box for the user to paste the key
        self.load_textbox = TextInputBox(100, 300, 400, 32)

    def generate_state_from_seed(self, seed_str):
        """Generate game state from any string seed."""
        # Use hashlib to get a consistent hash from the string
        seed_hash = hashlib.sha256(seed_str.encode('utf-8')).hexdigest()
        # Convert hash to an integer
        seed_int = int(seed_hash, 16) % 2**32  # Limit to 32 bits
        # Seed the random number generator
        np.random.seed(seed_int)
        # Generate logic_grid and cell_state_grid
        self.logic_grid = np.random.randint(0, len(RULESETS), size=self.grid_size, dtype=np.int32)
        self.cell_state_grid = np.random.randint(0, len(self.colors_list)-1, size=self.grid_size, dtype=np.int32)
        print(f"Generated game state from seed '{seed_str}'.")

class TextInputBox:
    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color_inactive = pygame.Color('lightskyblue3')
        self.color_active = pygame.Color('dodgerblue2')
        self.color = self.color_inactive
        self.text = text
        self.txt_surface = pygame.font.Font(None, 24).render(text, True, self.color)
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Toggle the active variable if the user clicked on the input_box rect
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            # Change the color of the input box
            self.color = self.color_active if self.active else self.color_inactive

        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    # Handle the return key
                    return self.text
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render the text
                self.txt_surface = pygame.font.Font(None, 24).render(self.text, True, self.color)
        return None

    def update(self):
        # Resize the box if the text is too long
        width = max(200, self.txt_surface.get_width()+10)
        self.rect.w = width

    def draw(self, screen):
        # Blit the text
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        # Blit the rect
        pygame.draw.rect(screen, self.color, self.rect, 2)

# Numba-optimized update function with color averaging
@njit
def update_cells(cell_state_grid, logic_grid, birth_rules_array, survival_rules_array, rule_lengths, black_index, neighbor_offsets, colors_array):
    rows, cols = cell_state_grid.shape
    new_grid = cell_state_grid.copy()
    num_colors = colors_array.shape[0]
    for i in range(rows):
        for j in range(cols):
            # Get the ruleset for the current cell
            ruleset_id = logic_grid[i, j]
            if ruleset_id == -1:
                continue  # Skip cells without logic assigned
            birth_rules = birth_rules_array[ruleset_id]
            survival_rules = survival_rules_array[ruleset_id]
            B_len = rule_lengths[ruleset_id, 0]
            S_len = rule_lengths[ruleset_id, 1]
            live_neighbors = 0
            color_sum = np.zeros(3, dtype=np.float64)
            for offset in neighbor_offsets:
                ni = (i + offset[0]) % rows
                nj = (j + offset[1]) % cols
                neighbor_index = cell_state_grid[ni, nj]
                if neighbor_index != black_index:
                    live_neighbors += 1
                    neighbor_color = colors_array[neighbor_index].astype(np.float64)
                    color_sum += neighbor_color
            is_alive = cell_state_grid[i, j] != black_index
            if is_alive:
                survived = False
                for k in range(S_len):
                    if live_neighbors == survival_rules[k]:
                        survived = True
                        break
                if not survived:
                    new_grid[i, j] = black_index  # Cell dies
                else:
                    if live_neighbors > 0:
                        avg_color = color_sum / live_neighbors
                        # Find the closest color index
                        min_diff = np.inf
                        closest_color_index = 0
                        for idx in range(num_colors - 1):  # Exclude black color
                            color_diff = ((colors_array[idx].astype(np.float64) - avg_color) ** 2).sum()
                            if color_diff < min_diff:
                                min_diff = color_diff
                                closest_color_index = idx
                        new_grid[i, j] = closest_color_index
            else:
                born = False
                for k in range(B_len):
                    if live_neighbors == birth_rules[k]:
                        born = True
                        break
                if born:
                    avg_color = color_sum / live_neighbors
                    # Find the closest color index
                    min_diff = np.inf
                    closest_color_index = 0
                    for idx in range(num_colors - 1):  # Exclude black color
                        color_diff = ((colors_array[idx].astype(np.float64) - avg_color) ** 2).sum()
                        if color_diff < min_diff:
                            min_diff = color_diff
                            closest_color_index = idx
                    new_grid[i, j] = closest_color_index
    return new_grid

if __name__ == "__main__":
    app = Factory_View()
    app.run()

# End of the line, partner.
