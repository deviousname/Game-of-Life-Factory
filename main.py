import pygame
import numpy as np
import colorsys
from numba import njit
import base64
import zlib
import pyperclip
import hashlib

# Serialization functions
def serialize_state(logic_grid, cell_state_grid, logic_inventory):
    # Convert grids to bytes
    logic_bytes = logic_grid.tobytes()
    cell_state_bytes = cell_state_grid.tobytes()

    # Serialize logic_inventory
    logic_ids = sorted(RULESET_IDS.values())
    logic_counts = [logic_inventory[logic_id] for logic_id in logic_ids]
    logic_counts_array = np.array(logic_counts, dtype=np.int32)
    logic_inventory_bytes = logic_counts_array.tobytes()

    # Concatenate bytes
    combined_bytes = logic_bytes + cell_state_bytes + logic_inventory_bytes

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

    # Calculate sizes
    grid_size = grid_shape[0] * grid_shape[1] * np.dtype(np.int32).itemsize
    logic_bytes_size = grid_size
    cell_state_bytes_size = grid_size

    total_bytes_needed = logic_bytes_size + cell_state_bytes_size

    if len(combined_bytes) >= total_bytes_needed:
        # Extract bytes
        logic_bytes = combined_bytes[:logic_bytes_size]
        cell_state_bytes = combined_bytes[logic_bytes_size:logic_bytes_size + cell_state_bytes_size]
        logic_inventory_bytes = combined_bytes[logic_bytes_size + cell_state_bytes_size:]

        # Convert bytes back to NumPy arrays
        logic_grid = np.frombuffer(logic_bytes, dtype=np.int32).reshape(grid_shape).copy()
        cell_state_grid = np.frombuffer(cell_state_bytes, dtype=np.int32).reshape(grid_shape).copy()

        # Recover logic_inventory if available
        num_rulesets = len(RULESET_IDS)
        expected_logic_inventory_size = num_rulesets * np.dtype(np.int32).itemsize
        if len(logic_inventory_bytes) >= expected_logic_inventory_size:
            logic_counts_array = np.frombuffer(logic_inventory_bytes[:expected_logic_inventory_size], dtype=np.int32)
            logic_ids = sorted(RULESET_IDS.values())
            logic_inventory = {logic_id: count for logic_id, count in zip(logic_ids, logic_counts_array)}
        else:
            # Logic inventory data not present, set to default
            logic_inventory = {ruleset_id: 0 for ruleset_id in RULESET_IDS.values()}
            total_cells = grid_shape[0] * grid_shape[1]
            logic_inventory[RULESET_IDS["Conway"]] = total_cells  # Start with Conway blocks
            # Include Void logic as infinite
            logic_inventory[RULESET_IDS["Void"]] = total_cells

    else:
        raise ValueError("Invalid key: not enough data")

    return logic_grid, cell_state_grid, logic_inventory

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

# Define primary colors in RGB space, adding white and orange
PRIMARY_COLORS = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (0, 255, 255),   # Cyan
    (255, 0, 255),   # Magenta
    (255, 165, 0),   # Orange
    (255, 255, 255)  # White
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
        grid_size=(32, 64),
        cell_size=32,
        margin=2,
        n_colors=64,
        window_title="Factory View",
        fullscreen=False, # True is buggy, scaling and text alignment issues
        fps_drawing=60,  # FPS for rendering the screen
        fps_simulation=60  # FPS for simulation updates
    ):
        
        # Grid settings
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.margin = margin
        
        # Initialize Pygame
        pygame.init()
        self.fullscreen = fullscreen
        if fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            display_width, display_height = self.screen.get_size()

            # Ensure the aspect ratio is maintained based on the grid size
            aspect_ratio = self.grid_size[1] / self.grid_size[0]  # width/height
            if display_width / display_height > aspect_ratio:
                # Limit by height
                self.cell_size = (display_height - margin) // self.grid_size[0]
            else:
                # Limit by width
                self.cell_size = (display_width - margin) // self.grid_size[1]

            # Calculate actual grid width and height
            self.width = self.cell_size * self.grid_size[1] + margin * (self.grid_size[1] + 1)
            self.height = self.cell_size * self.grid_size[0] + margin * (self.grid_size[0] + 1) + 30  # Extra space for UI

            # Create a surface for the grid and center it
            self.grid_surface = pygame.Surface((self.width, self.height))
        else:
            # Non-fullscreen logic remains unchanged
            self.width = grid_size[1] * (cell_size + margin) + margin
            self.height = grid_size[0] * (cell_size + margin) + margin + 30  # Extra space for UI
            self.screen = pygame.display.set_mode((self.width, self.height))

        pygame.display.set_caption(window_title)
        
        # Simulation and drawing FPS control
        self.fps_drawing = fps_drawing
        self.fps_simulation = fps_simulation
        self.simulation_time_accumulator = 0.0  # To track the simulation time
        self.simulation_interval = 1.0 / self.fps_simulation  # Time between each simulation update
        
        # Generate colors and adjust n_colors to account for black and white
        self.n_colors = n_colors
        self.colors_list = [
            tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, 1, 1))
            for h in np.linspace(0, 1, self.n_colors - 3, endpoint=False)
        ]
        # Append black, white, and dead cell color
        self.colors_list.append((0, 0, 0))          # Black color
        self.colors_list.append((255, 255, 255))    # White color
        self.dead_cell_color = (127, 127, 127)         # Dead cell color
        self.colors_list.append(self.dead_cell_color)

        # Set indices
        self.black_index = len(self.colors_list) - 3
        self.white_index = len(self.colors_list) - 2
        self.dead_cell_index = len(self.colors_list) - 1

        # Restructure the colors list by proximity to primary colors
        restructured_colors_list = restructure_colors(self.colors_list)

        # Convert to numpy array
        self.colors_array = np.array(restructured_colors_list, dtype=np.uint8)

        # Generate logic colors for better differentiation
        # Map logic IDs to colors
        self.logic_colors_list = {}
        logic_names = [name for name in RULESETS.keys() if name != 'Void']
        logic_ids = [RULESET_IDS[name] for name in logic_names]

        for idx, logic_name in enumerate(logic_names):
            logic_id = RULESET_IDS[logic_name]
            color = PRIMARY_COLORS[idx]
            self.logic_colors_list[logic_id] = color

        # Assign dark grey color to 'Void' logic
        void_logic_id = RULESET_IDS['Void']
        self.logic_colors_list[void_logic_id] = (50, 50, 50)  # Slightly dark grey

        # Map logic IDs to primary color indices
        self.logic_id_to_color_index = {logic_id: idx for idx, logic_id in enumerate(logic_ids)}

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
        self.mode = 'loading'  # Modes: 'building', 'simulation', 'menu', 'shop'
        self.previous_mode = 'building'
        self.paused = True
        
        # Other settings
        self.clock = pygame.time.Clock()
        self.done = False
        self.bonus = 0  # Player's energy or bonus points
        
        # Initialize player's logic block inventory
        self.logic_inventory = {ruleset_id: 0 for ruleset_id in RULESET_IDS.values()}
        # Infinite logic IDs (Conway and Void)
        self.infinite_logic_ids = {RULESET_IDS["Conway"], RULESET_IDS["Void"]}
        # Initialize Conway and Void with some value (infinite)
        total_cells = self.grid_size[0] * self.grid_size[1]
        self.logic_inventory[RULESET_IDS["Conway"]] = total_cells  # Start with Conway blocks
        self.logic_inventory[RULESET_IDS["Void"]] = total_cells  # Infinite Void blocks

        # Initialize prices for logic types
        self.logic_prices = {}
        available_logic_names = [name for name in RULESETS.keys() if name not in ('Void', 'Conway')]
        base_price = 1e1

        for idx, logic_name in enumerate(available_logic_names):
            price = int(base_price * (10 ** idx) + 0.5)
            logic_id = RULESET_IDS[logic_name]
            self.logic_prices[logic_id] = price

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

        # Additions for tallying colors and energy
        self.tally_frequency = 24  # Class variable to control tally frequency
        self.frame_counter = 0
        self.color_counts = np.zeros(len(PRIMARY_COLORS), dtype=int)
        self.energy = 0

        # For energy generation rate
        self.energy_generated_last = 0
        self.energy_generation_rate = 0.0
        self.energy_generation_timer = 0.0

        # NEW: Notation toggle
        self.scientific_notation = False  # Add this line to initialize the notation toggle

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

            if self.mode == 'loading':
                self.handle_loading()
                # Limit the frame rate during loading
                self.clock.tick(10)  # 10 FPS during loading
            else:
                # Existing code
                dt = self.clock.tick(self.fps_drawing) / 1000.0
                self.simulation_time_accumulator += dt

                if not self.paused and self.simulation_time_accumulator >= self.simulation_interval:
                    self.update()
                    self.simulation_time_accumulator -= self.simulation_interval

                self.draw()
                pygame.display.update()

    def handle_loading(self):
        """Display a loading screen and precompile Numba functions."""
        # Display loading screen
        self.screen.fill((0, 0, 0))  # Clear screen
        font = pygame.font.SysFont(None, 48)
        loading_text = font.render("Loading...", True, (255, 255, 255))
        text_rect = loading_text.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(loading_text, text_rect)
        pygame.display.update()

        # Pre-compile Numba functions by calling them with dummy data
        self.precompile_numba_functions()

        # After loading is done, switch to 'building' mode
        self.mode = 'building'
        self.paused = True  # Start paused or not, depending on desired behavior

    def precompile_numba_functions(self):
        """Precompile Numba functions by calling them with dummy data."""
        # Create dummy data matching the expected types
        dummy_cell_state_grid = np.zeros(self.grid_size, dtype=np.int32)
        dummy_logic_grid = np.zeros(self.grid_size, dtype=np.int32)
        dummy_birth_rules_array = self.birth_rules_array
        dummy_survival_rules_array = self.survival_rules_array
        dummy_rule_lengths = self.rule_lengths
        dummy_dead_cell_index = self.dead_cell_index
        dummy_neighbor_offsets = self.neighbor_offsets
        dummy_colors_array = self.colors_array  # Add this line

        # Call update_cells() with dummy data
        update_cells(
            dummy_cell_state_grid,
            dummy_logic_grid,
            dummy_birth_rules_array,
            dummy_survival_rules_array,
            dummy_rule_lengths,
            dummy_dead_cell_index,
            dummy_neighbor_offsets,
            dummy_colors_array  # Pass the colors array here
        )
    
    def cycle_ruleset(self, forward=True):
        """Cycle through only the rulesets you have blocks for."""
        # Get the list of available rulesets with non-zero blocks or infinite logic
        available_rulesets = [name for name, id in RULESET_IDS.items()
                              if self.logic_inventory[id] > 0 or id in self.infinite_logic_ids]

        # If there are no available rulesets, do nothing
        if not available_rulesets:
            return

        # If the current selected ruleset is not in the available list, reset to the first available one
        if self.selected_ruleset_name not in available_rulesets:
            self.selected_ruleset_name = available_rulesets[0]
            self.selected_ruleset_id = RULESET_IDS[self.selected_ruleset_name]
            return

        # Find the current ruleset index
        current_index = available_rulesets.index(self.selected_ruleset_name)

        # Move forward or backward in the list
        if forward:
            new_index = (current_index + 1) % len(available_rulesets)
        else:
            new_index = (current_index - 1) % len(available_rulesets)

        # Update the selected ruleset
        self.selected_ruleset_name = available_rulesets[new_index]
        self.selected_ruleset_id = RULESET_IDS[self.selected_ruleset_name]

    def handle_events(self):
        """Handle user input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
                pygame.quit()
                exit()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if self.mode == 'menu':
                    self.mode = self.previous_mode
                elif self.mode == 'shop':
                    self.mode = self.previous_mode  # Return to the correct mode
                else:
                    self.previous_mode = self.mode  # Save the current mode
                    self.mode = 'menu'
                    self.paused = True

            if self.mode == 'menu':
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        mouse_pos = pygame.mouse.get_pos()
                        if self.copy_button_rect.collidepoint(mouse_pos):
                            key = serialize_state(self.logic_grid, self.cell_state_grid, self.logic_inventory)
                            if self.clipboard_available:
                                pyperclip.copy(key)
                                print("Seed copied to clipboard.")
                            else:
                                print("Clipboard not available.")
                        if self.paste_button_rect.collidepoint(mouse_pos):
                            if self.clipboard_available:
                                try:
                                    clipboard_content = pyperclip.paste()
                                    try:
                                        self.logic_grid, self.cell_state_grid, self.logic_inventory = deserialize_state(clipboard_content, self.grid_size)
                                        print("Seed loaded from clipboard.")
                                    except Exception as e:
                                        print(f"Error loading seed: {e}")
                                        if clipboard_content.lower() == 'godmode':
                                            self.generate_state_from_seed(clipboard_content)
                                        else:
                                            print("Invalid seed. Cannot load game state.")
                                except Exception as e:
                                    print(f"Error accessing clipboard: {e}")
                            else:
                                print("Clipboard not available.")

            elif self.mode == 'shop':
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_b, pygame.K_ESCAPE):
                        self.mode = self.previous_mode  # Return to the correct mode
                    elif event.key == pygame.K_n:
                        self.scientific_notation = not self.scientific_notation  # Toggle notation
                    elif event.key == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        mouse_pos = pygame.mouse.get_pos()
                        for button_rect, logic_id, quantity in self.buy_buttons:
                            if button_rect.collidepoint(mouse_pos):
                                price_per_block = self.logic_prices[logic_id]
                                total_cost = price_per_block * quantity  # Buying quantity blocks
                                if self.bonus >= total_cost:
                                    self.bonus -= total_cost
                                    self.logic_inventory[logic_id] += quantity
                                    print(f"Purchased {quantity} blocks of {ID_RULESETS[logic_id]}")
                                else:
                                    print("Not enough energy to purchase.")
                                break
            else:
                # Handle events in other modes
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_TAB:
                        self.mode = 'simulation' if self.mode == 'building' else 'building'
                    elif event.key == pygame.K_SPACE and self.mode == 'simulation':
                        self.paused = not self.paused
                    elif event.key == pygame.K_d:
                        if self.mode == 'building':
                            self.cycle_ruleset(forward=True)
                        elif self.mode == 'simulation':
                            self.selected_color_index = (self.selected_color_index + 1) % (len(self.colors_list) - 1)
                    elif event.key == pygame.K_a:
                        if self.mode == 'building':
                            self.cycle_ruleset(forward=False)
                        elif self.mode == 'simulation':
                            self.selected_color_index = (self.selected_color_index - 1) % (len(self.colors_list) - 1)
                    elif event.key == pygame.K_f:
                        self.handle_flood_fill()
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_b:
                        # Save the current mode before switching to the buy menu
                        self.previous_mode = self.mode
                        self.mode = 'shop'
                    elif event.key == pygame.K_n:
                        self.scientific_notation = not self.scientific_notation  # Toggle notation
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
            # Refund logic blocks before resetting
            unique_logic_ids, counts = np.unique(self.logic_grid, return_counts=True)
            for logic_id, count in zip(unique_logic_ids, counts):
                if logic_id != 0 and logic_id not in self.infinite_logic_ids:
                    self.logic_inventory[logic_id] += count
            # Now reset the logic grid
            self.initialize_logic_grid()
            # Do not reset the player's inventory
            self.selected_ruleset_name = "Conway"  # Reset the selected ruleset to the default
            self.selected_ruleset_id = RULESET_IDS[self.selected_ruleset_name]
        elif self.mode == 'simulation':
            # Only reset the cell state grid for the simulation mode
            self.initialize_cell_state_grid()
            self.paused = True  # Optionally pause the simulation when resetting

    def handle_flood_fill(self):
        """Handle flood fill action based on mode."""
        pos = pygame.mouse.get_pos()
        if self.fullscreen:
            # Calculate grid surface position offset (since it's centered)
            grid_x_offset = (self.screen.get_width() - self.grid_surface.get_width()) // 2
            grid_y_offset = (self.screen.get_height() - self.grid_surface.get_height()) // 2

            # Adjust the mouse position to be relative to the grid surface
            adjusted_x = pos[0] - grid_x_offset
            adjusted_y = pos[1] - grid_y_offset - 30  # Adjust for UI height if needed

            col = adjusted_x // (self.cell_size + self.margin)
            row = adjusted_y // (self.cell_size + self.margin)
        else:
            col = pos[0] // (self.cell_size + self.margin)
            row = (pos[1] - 30) // (self.cell_size + self.margin)  # Adjust for UI height

        if 0 <= row < self.grid_size[0] and 0 <= col < self.grid_size[1]:
            if self.mode == 'building':
                target_logic = self.logic_grid[row, col]
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
        self.cell_state_grid = np.full((rows, cols), self.dead_cell_index, dtype=np.int32).copy()  # Ensure writable

    def flood_fill_logic(self, row, col, target_logic):
        """Flood fill logic for adjacent cells, including logicless cells (Void, 0)."""

        # Prevent flood-filling with Void logic over Void cells
        if self.selected_ruleset_id == RULESET_IDS["Void"] and self.logic_grid[row, col] == RULESET_IDS["Void"]:
            print("Cannot flood-fill with Void logic over Void cells.")
            return

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
                # Only fill cells that have the initial value
                if self.logic_grid[r, c] == initial_value:
                    # Check if player has blocks of the selected logic type or if it's infinite
                    if self.logic_inventory.get(self.selected_ruleset_id, 0) > 0 or self.selected_ruleset_id in self.infinite_logic_ids:
                        # Refund the old logic block if it's not Void and not infinite
                        old_logic_id = self.logic_grid[r, c]
                        if old_logic_id != RULESET_IDS["Void"] and old_logic_id not in self.infinite_logic_ids:
                            self.logic_inventory[old_logic_id] += 1
                        self.logic_grid[r, c] = self.selected_ruleset_id  # Apply selected ruleset
                        # Decrease inventory if not infinite
                        if self.selected_ruleset_id not in self.infinite_logic_ids:
                            self.logic_inventory[self.selected_ruleset_id] -= 1
                        neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
                        for nr, nc in neighbors:
                            if 0 <= nr < self.grid_size[0] and 0 <= nc < self.grid_size[1]:
                                if self.logic_grid[nr, nc] == initial_value:
                                    stack.append((nr, nc))
                    else:
                        print("Not enough blocks of this logic type.")
                        return

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

        # Adjust for fullscreen mode if necessary
        if self.fullscreen:
            # Calculate grid surface position offset (since it's centered)
            grid_x_offset = (self.screen.get_width() - self.grid_surface.get_width()) // 2
            grid_y_offset = (self.screen.get_height() - self.grid_surface.get_height()) // 2

            # Adjust the mouse position to be relative to the grid surface
            adjusted_x = pos[0] - grid_x_offset
            adjusted_y = pos[1] - grid_y_offset - 30  # Adjust for UI height if needed

            # Only proceed if the mouse is within the grid surface
            if 0 <= adjusted_x < self.grid_surface.get_width() and 0 <= adjusted_y < self.grid_surface.get_height():
                col = adjusted_x // (self.cell_size + self.margin)
                row = adjusted_y // (self.cell_size + self.margin)
            else:
                return  # Ignore clicks outside the grid surface
        else:
            # For non-fullscreen mode, no adjustment needed
            col = pos[0] // (self.cell_size + self.margin)
            row = (pos[1] - 30) // (self.cell_size + self.margin)  # Adjust for UI height

        # Handle painting in the grid based on the current mode
        if 0 <= row < self.grid_size[0] and 0 <= col < self.grid_size[1]:
            if self.mode == 'building':
                if self.mouse_buttons[0]:  # Left click to assign logic
                    if self.logic_inventory.get(self.selected_ruleset_id, 0) > 0 or self.selected_ruleset_id in self.infinite_logic_ids:
                        old_logic_id = self.logic_grid[row, col]
                        if old_logic_id != self.selected_ruleset_id:
                            # Refund the old logic block if it's not Void and not infinite
                            if old_logic_id != 0 and old_logic_id not in self.infinite_logic_ids:
                                self.logic_inventory[old_logic_id] += 1
                            self.logic_grid[row, col] = self.selected_ruleset_id
                            if self.selected_ruleset_id not in self.infinite_logic_ids:
                                self.logic_inventory[self.selected_ruleset_id] -= 1
                    else:
                        print("Not enough blocks of this logic type.")
                if self.mouse_buttons[2]:  # Right click to erase logic (set to Void state)
                    old_logic_id = self.logic_grid[row, col]
                    if old_logic_id != 0 and old_logic_id not in self.infinite_logic_ids:
                        self.logic_inventory[old_logic_id] += 1
                    self.logic_grid[row, col] = RULESET_IDS["Void"]  # Set back to Void logic
            if self.mode == 'simulation':
                if self.mouse_buttons[0]:  # Left click to paint living cells
                    self.cell_state_grid[row, col] = self.selected_color_index
                elif self.mouse_buttons[2]:  # Right click to erase cells
                    self.cell_state_grid[row, col] = self.dead_cell_index

    def update(self):
        """Update the simulation."""
        if self.mode == 'simulation' and not self.paused:
            self.cell_state_grid, births_count = update_cells(
                self.cell_state_grid,
                self.logic_grid,
                self.birth_rules_array,
                self.survival_rules_array,
                self.rule_lengths,
                self.dead_cell_index,
                self.neighbor_offsets,
                self.colors_array
            )

            # Increment frame counter
            self.frame_counter += 1

            # Accumulate births for energy generation
            self.energy_generated_last += births_count

            # Tally colors every 'tally_frequency' frames
            if self.frame_counter % self.tally_frequency == 0:
                self.tally_colors()  # Keep for color_counts and total living cells
                # Adjusted energy generation rate
                self.bonus += self.energy_generated_last * 1.0  # Reduced energy gain per birth

                # Additional bonus based on color prevalence
                for logic_id in self.logic_inventory.keys():
                    if logic_id in self.logic_id_to_color_index:
                        color_idx = self.logic_id_to_color_index[logic_id]
                        color_count = self.color_counts[color_idx]
                        bonus_for_logic = color_count // 100  # Example bonus calculation
                        self.bonus += bonus_for_logic * 0.1  # Reduced bonus

                self.energy_generation_timer += self.simulation_interval * self.tally_frequency

                if self.energy_generation_timer >= 1.0:
                    # Save the value of energy generated for display before resetting
                    energy_generated_for_display = self.energy_generated_last

                    # Calculate energy generation rate
                    self.energy_generation_rate = (energy_generated_for_display * 1.0) / self.energy_generation_timer

                    # Reset counters
                    self.energy_generated_last = 0  # Only reset after calculating generation rate
                    self.energy_generation_timer = 0.0

    def tally_colors(self):
        """Tally the living cells and their colors closest to primary colors."""
        self.color_counts = np.zeros(len(PRIMARY_COLORS), dtype=int)
        self.energy = 0  # Count of living cells

        rows, cols = self.cell_state_grid.shape
        for i in range(rows):
            for j in range(cols):
                color_index = self.cell_state_grid[i, j]
                if color_index != self.dead_cell_index:
                    self.energy += 1  # Count living cell
                    cell_color = self.colors_array[color_index]
                    # Find the closest primary color
                    min_distance = float('inf')
                    closest_primary_idx = -1
                    for idx, primary_color in enumerate(PRIMARY_COLORS):
                        distance = color_distance(cell_color, primary_color)
                        if distance < min_distance:
                            min_distance = distance
                            closest_primary_idx = idx
                    if closest_primary_idx != -1:
                        self.color_counts[closest_primary_idx] += 1

    def draw_menu(self):
        """Draw the menu with the previous mode as the background."""
        # Draw the background grid from the previous mode (either build or simulation)
        self.draw_game()

        # Draw the menu overlay on top
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
        font = pygame.font.Font(None, 28)
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
            
    def draw_shop(self):
        """Draw the buy menu with the current mode (building or simulation) as the background."""
        # Draw the background grid from the current mode (either build or simulation)
        self.draw_game()

        # Draw the semi-transparent overlay and buy menu on top
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # Semi-transparent black
        self.screen.blit(overlay, (0, 0))

        # Font for drawing text
        font = pygame.font.Font(None, 28)

        # Calculate the width of each column based on screen width
        left_column_x = self.width * 0.1  # Logic names (left aligned)
        center_column_x = self.width * 0.5  # Prices (center aligned)
        right_column_x = self.width * 0.9  # Buy buttons (right aligned)

        # Define padding between elements
        row_height = 40
        box_width = self.width * 0.8  # Width of the popup box
        box_height = 50 + row_height * len(self.logic_prices)  # Height based on number of logic types
        box_x = (self.width - box_width) // 2
        box_y = (self.height - box_height) // 2
        popup_rect = pygame.Rect(box_x, box_y, box_width, box_height)
        pygame.draw.rect(self.screen, (50, 50, 50), popup_rect)

        # Draw the player's current balance at the top
        balance_text = font.render(f"Current Energy: {self.format_number(self.bonus)}", True, (255, 255, 255))
        balance_rect = balance_text.get_rect(center=(self.width // 2, box_y + 20))
        self.screen.blit(balance_text, balance_rect)

        # List the logic types with prices and buy buttons
        y_offset = box_y + 60
        self.buy_buttons = []  # Store the rects of buy buttons for click detection

        for logic_id in sorted(self.logic_prices.keys()):
            logic_name = ID_RULESETS[logic_id]
            price_per_block = self.logic_prices[logic_id]

            # Logic Name (left aligned)
            logic_text = font.render(logic_name, True, (255, 255, 255))
            self.screen.blit(logic_text, (left_column_x, y_offset))

            # Price (center aligned)
            price_string = self.format_number(price_per_block)  # Use formatted price
            price_text = font.render(f"Price per block: {price_string}", True, (255, 255, 255))
            price_text_rect = price_text.get_rect(center=(center_column_x, y_offset + 10))  # Center align
            self.screen.blit(price_text, price_text_rect)

            # Buy 1 button (right aligned)
            buy1_button_rect = pygame.Rect(right_column_x - 190, y_offset - 5, 80, 30)
            pygame.draw.rect(self.screen, (70, 70, 70), buy1_button_rect)
            buy1_text = font.render('Buy 1', True, (255, 255, 255))
            buy1_text_rect = buy1_text.get_rect(center=buy1_button_rect.center)
            self.screen.blit(buy1_text, buy1_text_rect)

            # Buy 10 button (right aligned, to the right of Buy 1)
            buy10_button_rect = pygame.Rect(right_column_x - 100, y_offset - 5, 80, 30)
            pygame.draw.rect(self.screen, (70, 70, 70), buy10_button_rect)
            buy10_text = font.render('Buy 10', True, (255, 255, 255))
            buy10_text_rect = buy10_text.get_rect(center=buy10_button_rect.center)
            self.screen.blit(buy10_text, buy10_text_rect)

            # Store the rects and logic_id for click detection
            self.buy_buttons.append((buy1_button_rect, logic_id, 1))
            self.buy_buttons.append((buy10_button_rect, logic_id, 10))

            # Adjust y_offset for next row
            y_offset += row_height

    def draw(self):
        """Draw the grid and UI elements."""
        self.screen.fill((0, 0, 0))  # Clear screen

        if self.mode == 'menu':
            # Draw the game in the background
            self.draw_game()
            self.draw_menu()
        elif self.mode == 'shop':
            self.draw_game()
            self.draw_shop()
        else:
            self.draw_game()

    def draw_game(self):
        """Draw the game grid and UI based on the current mode."""
        # Decide which mode to use for drawing the grid
        if self.mode in ['building', 'simulation']:
            mode_to_draw = self.mode
        elif self.mode in ['shop', 'menu']:
            mode_to_draw = self.previous_mode
        else:
            mode_to_draw = self.mode  # default

        # Then, in the drawing code, use mode_to_draw instead of self.mode
        if self.fullscreen:
            # Draw to the smaller centered surface
            self.grid_surface.fill((0, 0, 0))

            # Draw the grid based on mode_to_draw
            for row in range(self.grid_size[0]):
                for col in range(self.grid_size[1]):
                    x = col * (self.cell_size + self.margin) + self.margin
                    y = row * (self.cell_size + self.margin) + self.margin + 30  # Adjust for UI height
                    if mode_to_draw == 'building':
                        # In building mode, draw logic grid
                        ruleset_id = self.logic_grid[row, col]
                        color = self.logic_colors_list.get(ruleset_id, (50, 50, 50))
                    else:
                        # In simulation mode, draw cell state grid
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
            # Draw the grid directly on the screen based on mode_to_draw
            for row in range(self.grid_size[0]):
                for col in range(self.grid_size[1]):
                    x = col * (self.cell_size + self.margin) + self.margin
                    y = row * (self.cell_size + self.margin) + self.margin + 30  # Adjust for UI height
                    if mode_to_draw == 'building':
                        # In building mode, draw logic grid
                        ruleset_id = self.logic_grid[row, col]
                        color = self.logic_colors_list.get(ruleset_id, (50, 50, 50))
                    else:
                        # In simulation mode, draw cell state grid
                        color_index = self.cell_state_grid[row, col]
                        color = self.colors_array[color_index]
                    pygame.draw.rect(
                        self.screen,
                        color,
                        (x, y, self.cell_size, self.cell_size)
                    )

        # UI for simulation mode
        font = pygame.font.SysFont(None, 28)
        font_large = pygame.font.SysFont(None, 48)
        
        if self.mode == 'simulation':
            # Display Mode: Simulation in the top-left corner
            mode_text_surface = render_text_with_outline(f"Mode: Simulation", font, (255, 255, 255), (0, 0, 0))
            self.screen.blit(mode_text_surface, (10, 5))

            # Start position for the "Alive:" text
            alive_text = f"Alive: {self.energy}"
            alive_text_surface = render_text_with_outline(alive_text, font, (255, 255, 255), (0, 0, 0))
            color_tally_x_position = self.width - 10  # Start from the right side

            # Subtract space for color counters
            for idx, color in enumerate(PRIMARY_COLORS):
                color_count = self.color_counts[idx]
                color_text_surface = render_text_with_outline(str(color_count), font, color, (0, 0, 0))
                color_tally_x_position -= color_text_surface.get_width() + 10  # Move left for each color tally
                self.screen.blit(color_text_surface, (color_tally_x_position, 5))

            # Now place the "Alive:" text to the left of the first color counter
            alive_text_x_position = color_tally_x_position - alive_text_surface.get_width() - 10
            self.screen.blit(alive_text_surface, (alive_text_x_position, 5))
            if self.paused:
                # Display "Let's Paint!!" in colorful text when paused
                text = "Let's Paint!!"
                text_surfaces = []
                for i, char in enumerate(text):
                    color = PRIMARY_COLORS[i % len(PRIMARY_COLORS)]
                    char_surface = render_text_with_outline(char, font_large, color, (0, 0, 0))
                    text_surfaces.append(char_surface)

                # Calculate total width of the "Let's Paint!!" text
                total_width = sum(surface.get_width() for surface in text_surfaces)
                x = (self.width - total_width) // 2
                y = 5  # Adjust y position if needed
                for surface in text_surfaces:
                    self.screen.blit(surface, (x, y))
                    x += surface.get_width()

            else:
                # Display "Game of Life: Factory" in alternating colors when unpaused
                text = "Game of Life: Factory"
                fancy_colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0)]  # Cyan, Magenta, Yellow
                text_surfaces = []
                for i, char in enumerate(text):
                    color = fancy_colors[i % len(fancy_colors)]
                    char_surface = render_text_with_outline(char, font_large, color, (0, 0, 0))
                    text_surfaces.append(char_surface)

                # Calculate total width of the "Game of Life: Factory" text
                total_width = sum(surface.get_width() for surface in text_surfaces)
                x = (self.width - total_width) // 2
                y = 5  # Adjust y position if needed
                for surface in text_surfaces:
                    self.screen.blit(surface, (x, y))
                    x += surface.get_width()

            # Display energy and alive cell counts below the text (both paused and unpaused)
            bonus_text = f"Energy: {self.format_number(self.bonus)} (+{self.format_number(self.energy_generation_rate)}/s)"
            bonus_surface = render_text_with_outline(bonus_text, font, (255, 255, 255), (0, 0, 0))
            x = (self.width - bonus_surface.get_width()) // 2
            self.screen.blit(bonus_surface, (x, 35))  # Adjust y position to not overlap

            # Then draw other simulation UI elements
            selected_color = self.colors_list[self.selected_color_index]
            color_text_surface = render_text_with_outline(f"Selected Color: {selected_color}", font, (255, 255, 255), (0, 0, 0))
            self.screen.blit(color_text_surface, (200, 5))
            instructions = "Tab: Switch Mode | A and D: Change Color | R: Reset  | Space: Pause | B: Buy | F: Fill Bucket | N: Notation"

        else:
            # Building mode UI
            mode_text_surface = render_text_with_outline(f"Mode: {self.mode.capitalize()}", font, (255, 255, 255), (0, 0, 0))
            self.screen.blit(mode_text_surface, (10, 5))
            # Building mode UI
            ruleset_text_surface = render_text_with_outline(f"Selected Ruleset: {self.selected_ruleset_name}", font, (255, 255, 255), (0, 0, 0))
            self.screen.blit(ruleset_text_surface, (200, 5))
            instructions = "Tab: Switch Mode | A and D: Change Ruleset | B: Buy | R: Reset | F: Fill Bucket | N: Notation"

            # Display logic inventory
            stats_font = pygame.font.SysFont(None, 28)
            x_position = self.width - 10  # 10 pixels padding from the right

            # Prepare logic IDs in the desired order
            logic_ids_order = [RULESET_IDS["Conway"]] + sorted(
                [ruleset_id for ruleset_name, ruleset_id in RULESET_IDS.items() if ruleset_name not in ("Conway", "Void")]
            ) + [RULESET_IDS["Void"]]

            for logic_id in reversed(logic_ids_order):
                logic_name = ID_RULESETS[logic_id]
                count_text = "∞" if logic_id in self.infinite_logic_ids else str(self.logic_inventory[logic_id])
                text_surface = render_text_with_outline(f"{logic_name}: {count_text}", stats_font, self.logic_colors_list.get(logic_id, (255, 255, 255)), (0, 0, 0))
                x_position -= text_surface.get_width()
                self.screen.blit(text_surface, (x_position, 5))
                x_position -= 10  # Move position for next text

        # Draw instructions at the bottom
        instructions_surface = render_text_with_outline(instructions, font, (255, 255, 255), (0, 0, 0))
        self.screen.blit(instructions_surface, (10, self.height - 25))

    def handle_save(self):
        # Generate the key
        key = serialize_state(self.logic_grid, self.cell_state_grid, self.logic_inventory)

        # Create a text input box with the key
        self.save_textbox = TextInputBox(100, 300, 400, 32, text=key)

    def handle_load(self):
        # Create an empty text input box for the user to paste the key
        self.load_textbox = TextInputBox(100, 300, 400, 32)

    def generate_state_from_seed(self, seed_str):
        """Generate game state from any string seed."""
        if seed_str.lower() == 'godmode':
            # Activate godmode
            total_cells = self.grid_size[0] * self.grid_size[1]
            self.logic_inventory = {ruleset_id: total_cells for ruleset_id in RULESET_IDS.values()}
            self.logic_grid = np.full(self.grid_size, fill_value=0, dtype=np.int32)
            self.cell_state_grid = np.full(self.grid_size, self.dead_cell_index, dtype=np.int32)
            print("Godmode activated: All logic blocks unlocked.")
        else:
            # Use hashlib to get a consistent hash from the string
            seed_hash = hashlib.sha256(seed_str.encode('utf-8')).hexdigest()
            # Convert hash to an integer
            seed_int = int(seed_hash, 16) % 2**32  # Limit to 32 bits
            # Seed the random number generator
            np.random.seed(seed_int)
            # Generate logic_grid and cell_state_grid
            self.logic_grid = np.random.randint(0, len(RULESETS), size=self.grid_size, dtype=np.int32)
            self.cell_state_grid = np.random.randint(0, len(self.colors_list)-1, size=self.grid_size, dtype=np.int32)
            # Generate random logic_inventory
            self.logic_inventory = {ruleset_id: np.random.randint(0, total_cells) for ruleset_id in RULESET_IDS.values()}
            print("Generated game state from seed.")

    def format_number(self, number):
        """Helper function to format numbers based on notation preference."""
        if self.scientific_notation:
            return f"{number:.1e}"
        else:
            return f"{int(number)}"

class TextInputBox:
    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color_inactive = pygame.Color('lightskyblue3')
        self.color_active = pygame.Color('dodgerblue2')
        self.color = self.color_inactive
        self.text = text
        self.txt_surface = pygame.font.Font(None, 28).render(text, True, self.color)
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
                self.txt_surface = pygame.font.Font(None, 28).render(self.text, True, self.color)
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
        
def render_text_with_outline(text, font, text_color, outline_color):
    # First, render the text normally
    text_surface = font.render(text, True, text_color)
    # Then, create a new surface slightly larger to accommodate the outline
    size = text_surface.get_width() + 2, text_surface.get_height() + 2
    outline_surface = pygame.Surface(size, pygame.SRCALPHA)
    # Draw the outline by rendering the text multiple times with offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
               (0, 1), (1, -1), (1, 0), (1, 1)]
    for dx, dy in offsets:
        pos = (1 + dx, 1 + dy)
        outline_surface.blit(font.render(text, True, outline_color), pos)
    # Blit the main text onto the outline surface
    outline_surface.blit(text_surface, (1, 1))
    return outline_surface

@njit
def shift_grid(grid, shift_row, shift_col):
    """Shift the grid manually along both axes."""
    rows, cols = grid.shape
    result = np.zeros_like(grid)
    
    if shift_row > 0:
        result[shift_row:, :] = grid[:-shift_row, :]
    elif shift_row < 0:
        result[:shift_row, :] = grid[-shift_row:, :]
    else:
        result[:, :] = grid[:, :]
    
    if shift_col > 0:
        result[:, shift_col:] = result[:, :-shift_col]
    elif shift_col < 0:
        result[:, :shift_col] = result[:, -shift_col:]
    
    return result

@njit
def convolve(grid, offsets):
    """Helper function to compute neighbor sums via manual shifting."""
    rows, cols = grid.shape
    neighbor_count = np.zeros((rows, cols), dtype=np.int32)

    for offset in offsets:
        neighbor_count += shift_grid(grid, offset[0], offset[1])

    return neighbor_count

@njit
def compute_color_sum(grid, colors_array, offsets, black_index):
    """Accumulate the color values for each cell's neighbors."""
    rows, cols = grid.shape
    color_sum = np.zeros((rows, cols, 3), dtype=np.float64)  # RGB sum for each cell

    for offset in offsets:
        shifted_grid = shift_grid(grid, offset[0], offset[1])
        for i in range(rows):
            for j in range(cols):
                if shifted_grid[i, j] != black_index:  # If neighbor is alive
                    color_sum[i, j] += colors_array[shifted_grid[i, j]]  # Accumulate color

    return color_sum

@njit
def update_cells(cell_state_grid, logic_grid, birth_rules_array, survival_rules_array, rule_lengths, black_index, neighbor_offsets, colors_array):
    rows, cols = cell_state_grid.shape
    new_grid = cell_state_grid.copy()
    num_colors = colors_array.shape[0]
    births_count = 0  # New variable to count births

    # 1. Precompute neighbor counts using manual shifting
    live_neighbor_count = convolve(cell_state_grid != black_index, neighbor_offsets)

    # 2. Precompute color sums for living neighbors
    neighbor_color_sum = compute_color_sum(cell_state_grid, colors_array, neighbor_offsets, black_index)

    # 3. Iterate through each cell and apply rules
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

            live_neighbors = live_neighbor_count[i, j]
            is_alive = cell_state_grid[i, j] != black_index

            if is_alive:
                # Check survival condition
                survived = False
                for k in range(S_len):
                    if live_neighbors == survival_rules[k]:
                        survived = True
                        break
                if not survived:
                    new_grid[i, j] = black_index  # Cell dies
                else:
                    # Calculate the average color of the living neighbors
                    if live_neighbors > 0:
                        avg_color = neighbor_color_sum[i, j] / live_neighbors
                        new_grid[i, j] = find_closest_color(colors_array, avg_color, num_colors)
            else:
                # Check birth condition
                born = False
                for k in range(B_len):
                    if live_neighbors == birth_rules[k]:
                        born = True
                        break
                if born and live_neighbors > 0:
                    avg_color = neighbor_color_sum[i, j] / live_neighbors
                    new_grid[i, j] = find_closest_color(colors_array, avg_color, num_colors)
                    births_count += 1  # Increment births count

    return new_grid, births_count

@njit
def find_closest_color(colors_array, avg_color, num_colors):
    """Find the closest color in the palette based on the average color."""
    min_diff = np.inf
    closest_color_index = 0

    for idx in range(num_colors - 1):  # Exclude black color
        color_diff = np.sum((colors_array[idx].astype(np.float64) - avg_color) ** 2)
        if color_diff < min_diff:
            min_diff = color_diff
            closest_color_index = idx

    return closest_color_index

if __name__ == "__main__":
    app = Factory_View()
    app.run()
