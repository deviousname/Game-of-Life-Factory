import pygame
import numpy as np
import colorsys
from numba import njit
import base64
import zlib
import pyperclip
import hashlib

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

# Create a NumPy array for base energy, indexed by the logic ID
logic_base_energy_array = np.zeros(len(RULESETS), dtype=np.int32)

# Set base energy values for each logic type
logic_base_energy_array[RULESET_IDS["Conway"]] = 1
logic_base_energy_array[RULESET_IDS["HighLife"]] = 2
logic_base_energy_array[RULESET_IDS["DayAndNight"]] = 3
logic_base_energy_array[RULESET_IDS["Seeds"]] = 4
logic_base_energy_array[RULESET_IDS["LifeWithoutDeath"]] = 5
logic_base_energy_array[RULESET_IDS["Maze"]] = 6
logic_base_energy_array[RULESET_IDS["Gnarl"]] = 7
logic_base_energy_array[RULESET_IDS["Replicator"]] = 8
logic_base_energy_array[RULESET_IDS["Void"]] = 0  # Void generates no energy

# Define primary colors in RGB space, adding pure white and black
PRIMARY_COLORS = [
    (255, 0, 0),     # Red (Fire)
    (0, 0, 255),     # Blue (Water)
    (0, 255, 0),     # Green (Bio)
    (255, 255, 0),   # Yellow (Lightning)
    (0, 255, 255),   # Cyan (Ice)
    (255, 0, 255),   # Magenta (Energy)
    (255, 255, 255), # White (Armor)
    (0, 0, 0)        # Black (True Damage)
]

ELEMENTAL_NAMES = [
    "Fire",
    "Water",
    "Bio",
    "Lightning",
    "Ice",
    "Energy",
    "Armor",
    "True Damage"
]

class Factory:
    def __init__(
        self,
        grid_size=(42, 84),
        cell_size=21,
        margin=5,
        n_colors=64,
        window_title="Battlecells",
        fullscreen=False,
        fps_drawing=60,
        fps_simulation=60
    ):
        # Initialize core settings
        self.init_core_settings(grid_size, cell_size, margin, n_colors, window_title, fullscreen, fps_drawing, fps_simulation)

        # Initialize pygame and set up display
        self.init_pygame(fullscreen)

        # Color management setup
        self.setup_colors()

        # Logic color mapping
        self.setup_logic_colors()

        # Initialize gameplay-related settings
        self.setup_gameplay()

        # Initialize player's logic inventory and pricing
        self.setup_logic_inventory()

        # Initialize additional game variables
        self.setup_misc()

        # Initialize clipboard support
        self.init_clipboard()

        # Modes and pause state
        self.mode = 'loading'
        self.previous_mode = 'building'
        self.paused = True

    def init_core_settings(self, grid_size, cell_size, margin, n_colors, window_title, fullscreen, fps_drawing, fps_simulation):
        """Initialize core settings for the game."""
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.margin = margin
        self.n_colors = n_colors
        self.window_title = window_title
        self.fullscreen = fullscreen
        self.fps_drawing = fps_drawing
        self.fps_simulation = fps_simulation
        self.simulation_time_accumulator = 0.0
        self.simulation_interval = 1.0 / fps_simulation

    def init_pygame(self, fullscreen):
        """Initialize pygame and set up the display."""
        pygame.init()
        if fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.adjust_for_fullscreen()
        else:
            self.width = self.grid_size[1] * (self.cell_size + self.margin) + self.margin
            self.height = self.grid_size[0] * (self.cell_size + self.margin) + self.margin + 30  # Extra space for UI
            self.screen = pygame.display.set_mode((self.width, self.height))
        
        pygame.display.set_caption(self.window_title)

    def adjust_for_fullscreen(self):
        """Adjust grid size and display for fullscreen mode."""
        display_width, display_height = self.screen.get_size()
        aspect_ratio = self.grid_size[1] / self.grid_size[0]
        if display_width / display_height > aspect_ratio:
            self.cell_size = (display_height - self.margin) // self.grid_size[0]
        else:
            self.cell_size = (display_width - self.margin) // self.grid_size[1]

        self.width = self.cell_size * self.grid_size[1] + self.margin * (self.grid_size[1] + 1)
        self.height = self.cell_size * self.grid_size[0] + self.margin * (self.grid_size[0] + 1) + 30
        self.grid_surface = pygame.Surface((self.width, self.height))

    def setup_colors(self):
        """Generate and restructure colors for the grid."""
        self.n_colors -= 2  # Adjust for black and white added later
        self.colors_list = [
            tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, 1, 1))
            for h in np.linspace(0, 1, self.n_colors, endpoint=False)
        ]
        self.colors_list += [(0, 0, 0), (255, 255, 255), (127, 127, 127)]  # Black, White, Dead cell (Gray)
        self.black_index, self.white_index, self.dead_cell_index = len(self.colors_list) - 3, len(self.colors_list) - 2, len(self.colors_list) - 1
        restructured_colors_list = self.restructure_colors(self.colors_list)
        self.colors_array = np.array(restructured_colors_list, dtype=np.uint8)

    def setup_logic_colors(self):
        """Set up logic color mapping based on rulesets."""
        self.logic_colors_list = {}
        logic_names = [name for name in RULESETS.keys() if name != 'Void']
        for idx, logic_name in enumerate(logic_names):
            logic_id = RULESET_IDS[logic_name]
            self.logic_colors_list[logic_id] = PRIMARY_COLORS[idx]

        void_logic_id = RULESET_IDS['Void']
        self.logic_colors_list[void_logic_id] = (127, 127, 127)  # Dark grey for 'Void'

        self.logic_id_to_color_index = {RULESET_IDS[name]: idx for idx, name in enumerate(logic_names)}

    def setup_gameplay(self):
        """Initialize gameplay-related settings."""
        self.selected_color_index = self.black_index
        self.selected_ruleset_name = "Conway"
        self.selected_ruleset_id = RULESET_IDS[self.selected_ruleset_name]
        self.initialize_grids()
        self.logic_grid_history = []
        self.cell_state_grid_history = []
        self.max_history_length = 50
        self.preprocess_rulesets()
        self.clock = pygame.time.Clock()
        self.done = False

    def setup_logic_inventory(self):
        """Set up logic block inventory and pricing for the player."""
        self.logic_inventory = {ruleset_id: {'count': 0, 'tier': 1} for ruleset_id in RULESET_IDS.values()}
        
        # Infinite logic IDs (Conway and Void)
        self.infinite_logic_ids = {RULESET_IDS["Conway"], RULESET_IDS["Void"]}

        total_cells = self.grid_size[0] * self.grid_size[1]
        self.logic_inventory[RULESET_IDS["Conway"]]['count'] = total_cells
        self.logic_inventory[RULESET_IDS["Void"]]['count'] = total_cells

        self.logic_prices = {}
        base_price = 10
        available_logic_names = [name for name in RULESETS.keys() if name not in ('Void', 'Conway')]
        for i, logic_name in enumerate(available_logic_names):
            logic_id = RULESET_IDS[logic_name]
            self.logic_prices[logic_id] = base_price * (10 ** i)

        self.logic_prices[RULESET_IDS["Conway"]] = 0
        self.logic_prices[RULESET_IDS["Void"]] = 0

    def setup_misc(self):
        """Set up miscellaneous settings and variables."""
        self.bonus = 0
        self.bonus_timer = 0.0
        self.bonus_interval = 1.0
        self.minute_cell_changes = 0
        self.minute_color_counts = np.zeros(len(PRIMARY_COLORS), dtype=int)
        self.bonus_energy_rate = 0.0
        self.bonus_defense = 0.0
        self.bonus_offense = 0.0
        self.diversity_bonus = 0.0
        self.painted_cells_during_pause = set()

        self.neighbor_offsets = np.array([
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),         (0, 1),
            (1, -1), (1, 0), (1, 1)
        ], dtype=np.int32)

        self.mouse_buttons = [False, False, False]

        self.tally_frequency = 24
        self.frame_counter = 0
        self.color_counts = np.zeros(len(PRIMARY_COLORS), dtype=int)
        self.energy = 0

        self.core_energy_generated_last = 0
        self.bonus_energy_rate = 0.0
        self.total_energy_generated = 0
        self.energy_generation_timer = 0.0
        self.energy_generation_rate = 0

        self.scientific_notation = False
        self.minute_new_cells = 0
        self.minute_static_cells = 0
        self.defense = 0.0
        self.offense = 0.0
        self.fill_patterns = {
            0: [(0, 1), (1, 0), (0, -1), (-1, 0)],  # Right, Down, Left, Up
            1: [(1, 1), (-1, 1), (1, -1), (-1, -1)]  # Diagonals
        }
        self.current_pattern_index = 0

    def init_clipboard(self):
        """Initialize clipboard support for copying data."""
        try:
            import pyperclip
            self.clipboard_available = True
        except ImportError:
            self.clipboard_available = False
            
    # Serialization functions
    def serialize_state(self, logic_grid, cell_state_grid, logic_inventory, energy):
        # Convert grids to bytes
        logic_bytes = logic_grid.tobytes()
        cell_state_bytes = cell_state_grid.tobytes()

        # Serialize logic_inventory
        logic_ids = sorted(RULESET_IDS.values())
        logic_counts = [logic_inventory[logic_id]['count'] for logic_id in logic_ids]
        logic_tiers = [logic_inventory[logic_id]['tier'] for logic_id in logic_ids]
        logic_inventory_array = np.array(logic_counts + logic_tiers, dtype=np.int32)
        logic_inventory_bytes = logic_inventory_array.tobytes()

        # Serialize energy as a float (to preserve decimals)
        energy_bytes = np.array([energy], dtype=np.float32).tobytes()

        # Concatenate bytes
        combined_bytes = logic_bytes + cell_state_bytes + logic_inventory_bytes + energy_bytes

        # Compress the bytes
        compressed_bytes = zlib.compress(combined_bytes)

        # Encode as base64 to get a string
        key = base64.b64encode(compressed_bytes).decode('utf-8')
        return key

    def deserialize_state(self, key, grid_shape):
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
            logic_inventory_bytes = combined_bytes[logic_bytes_size + cell_state_bytes_size:-4]  # Leave last 4 bytes for energy
            energy_bytes = combined_bytes[-4:]  # The last 4 bytes are for the energy value

            # Convert bytes back to NumPy arrays
            logic_grid = np.frombuffer(logic_bytes, dtype=np.int32).reshape(grid_shape).copy()
            cell_state_grid = np.frombuffer(cell_state_bytes, dtype=np.int32).reshape(grid_shape).copy()

            # Recover logic_inventory
            num_rulesets = len(RULESET_IDS)
            expected_logic_inventory_size = num_rulesets * 2 * np.dtype(np.int32).itemsize
            if len(logic_inventory_bytes) >= expected_logic_inventory_size:
                logic_inventory_array = np.frombuffer(logic_inventory_bytes[:expected_logic_inventory_size], dtype=np.int32)
                logic_counts = logic_inventory_array[:num_rulesets]
                logic_tiers = logic_inventory_array[num_rulesets:num_rulesets*2]
                logic_ids = sorted(RULESET_IDS.values())
                logic_inventory = {
                    logic_id: {'count': count, 'tier': tier}
                    for logic_id, count, tier in zip(logic_ids, logic_counts, logic_tiers)
                }
            else:
                logic_inventory = {ruleset_id: {'count': 0, 'tier': 1} for ruleset_id in RULESET_IDS.values()}

            # Recover the energy value
            energy = np.frombuffer(energy_bytes, dtype=np.float32)[0]

        else:
            raise ValueError("Invalid key: not enough data")

        return logic_grid, cell_state_grid, logic_inventory, energy


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
                self.clock.tick(10)
            else:
                dt = self.clock.tick(self.fps_drawing) / 1000.0
                self.simulation_time_accumulator += dt

                if not self.paused and self.simulation_time_accumulator >= self.simulation_interval:
                    self.update()
                    self.simulation_time_accumulator -= self.simulation_interval

                self.draw()
                pygame.display.flip()

    def handle_loading(self):
        """Display a loading screen and precompile Numba functions."""
        #self.screen.fill((0, 0, 0))
        font = pygame.font.SysFont(None, 48)
        loading_text = font.render("Loading...", True, (255, 255, 255))
        text_rect = loading_text.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(loading_text, text_rect)
        pygame.display.flip()

        # Pre-compile Numba functions by calling them with dummy data
        self.precompile_numba_functions()

        # After loading is done, switch to 'building' mode
        self.mode = 'building'
        self.paused = True

    def precompile_numba_functions(self):
        """Precompile Numba functions by calling them with dummy data."""
        dummy_cell_state_grid = np.zeros(self.grid_size, dtype=np.int32)
        dummy_logic_grid = np.zeros(self.grid_size, dtype=np.int32)
        dummy_birth_rules_array = self.birth_rules_array
        dummy_survival_rules_array = self.survival_rules_array
        dummy_rule_lengths = self.rule_lengths
        dummy_black_index = self.dead_cell_index
        dummy_neighbor_offsets = self.neighbor_offsets
        dummy_colors_array = self.colors_array
        dummy_logic_base_energy_array = logic_base_energy_array
        dummy_logic_inventory_array = np.zeros((len(RULESETS), 2), dtype=np.int32)

        update_cells(
            dummy_cell_state_grid,
            dummy_logic_grid,
            dummy_birth_rules_array,
            dummy_survival_rules_array,
            dummy_rule_lengths,
            dummy_black_index,
            dummy_neighbor_offsets,
            dummy_colors_array,
            dummy_logic_inventory_array,
            dummy_logic_base_energy_array
        )

    def cycle_ruleset(self, forward=True):
        """Cycle through only the rulesets you have blocks for."""
        # Get the list of available rulesets with non-zero blocks or infinite logic
        available_rulesets = [name for name, id in RULESET_IDS.items()
                              if self.logic_inventory[id]['count'] > 0 or id in self.infinite_logic_ids]

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
            # Handle global events first
            if event.type == pygame.QUIT:
                self.done = True
                pygame.quit()
                exit()

            # Handle keydown events
            if event.type == pygame.KEYDOWN:
                self.handle_keydown(event)

            # Handle mouse events based on mode
            if self.mode == 'menu':
                self.handle_menu_mouse_event(event)
            elif self.mode == 'shop':
                self.handle_shop_mouse_event(event)
            else:
                self.handle_general_mouse_event(event)

    def handle_keydown(self, event):
        """Handle keydown events based on the current mode."""
        if event.key == pygame.K_ESCAPE:
            self.handle_escape_key()

        elif event.key == pygame.K_i:
            self.toggle_info_panel()
            
        elif event.key == pygame.K_l:
            self.print_living_cells()
            
        elif event.key == pygame.K_TAB:
            self.toggle_mode('simulation', 'building')

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

        elif event.key == pygame.K_w and self.mode == 'simulation':
            self.selected_color_index = self.black_index

        elif event.key == pygame.K_f:
            self.handle_flood_fill()

        elif event.key == pygame.K_r:
            self.reset_game()

        elif event.key == pygame.K_b:
            self.toggle_shop()

        elif event.key == pygame.K_n:
            self.scientific_notation = not self.scientific_notation

        elif event.key == pygame.K_t:
            self.increment_selected_logic_tier()

        elif event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL:
            self.undo_action()

    def handle_escape_key(self):
        """Handle the escape key behavior."""
        if self.mode in ['menu', 'shop', 'stats']:
            self.close_current_menu()
        else:
            self.open_menu('menu')  # Open copy/paste menu when no other menu is open
            
    def close_current_menu(self):
        """Close the current menu and return to the previous mode."""
        if self.mode in ['menu', 'shop', 'stats']:
            self.mode = self.previous_mode  # Go back to the previous mode
            self.paused = True  # Keep the game paused when exiting a menu

    def open_menu(self, new_mode):
        """Open a new menu, saving the current mode."""
        if self.mode not in ['menu', 'shop', 'stats']:
            self.previous_mode = self.mode  # Save the current mode before switching
        self.mode = new_mode
        self.paused = True  # Pause the game when opening a menu
    
    def toggle_info_panel(self):
        """Toggle the stats (info) panel with the 'i' key."""
        if self.mode == 'stats':
            self.close_current_menu()  # Close stats and return to previous mode
        else:
            if self.mode not in ['menu', 'shop', 'stats']:
                self.previous_mode = self.mode  # Save the current mode
            self.mode = 'stats'  # Open the stats panel without changing the game mode
            self.paused = True

    def toggle_mode(self, mode1, mode2):
        """Toggle between two modes (e.g., 'simulation' and 'building')."""
        self.mode = mode1 if self.mode == mode2 else mode2

    def toggle_shop(self):
        """Toggle the shop mode."""
        if self.mode == 'shop':
            self.close_current_menu()  # Close shop and return to previous mode
        else:
            self.open_menu('shop')

    def increment_selected_logic_tier(self):
        """Increase the tier of the selected logic block."""
        self.logic_inventory[self.selected_ruleset_id]['tier'] += 1
        print(f"{ID_RULESETS[self.selected_ruleset_id]} tier increased to {self.logic_inventory[self.selected_ruleset_id]['tier']}")

    def handle_menu_mouse_event(self, event):
        """Handle mouse events in menu mode."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
            mouse_pos = pygame.mouse.get_pos()
            if self.copy_button_rect.collidepoint(mouse_pos):
                self.handle_save()
            elif self.paste_button_rect.collidepoint(mouse_pos):
                self.handle_load()

    def handle_shop_mouse_event(self, event):
        """Handle mouse events in shop mode."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
            mouse_pos = pygame.mouse.get_pos()
            for button_rect, logic_id, quantity in self.buy_buttons:
                if button_rect.collidepoint(mouse_pos):
                    self.purchase_logic_block(logic_id, quantity)
                    break

    def handle_general_mouse_event(self, event):
        """Handle general mouse events."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button <= 3:
                self.mouse_buttons[event.button - 1] = True
            self.handle_mouse_event()
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button <= 3:
                self.mouse_buttons[event.button - 1] = False
        elif event.type == pygame.MOUSEMOTION:
            if any(self.mouse_buttons):
                self.handle_mouse_event()

    def calculate_max_purchase(self, logic_id):
        """Calculate the maximum number of tiles the player can buy for a logic type."""
        total_cells = self.grid_size[0] * self.grid_size[1]
        owned_blocks = self.logic_inventory[logic_id]['count']
        max_possible = total_cells - owned_blocks
        return max_possible

    def check_for_tier_upgrade(self, logic_id):
        """Upgrade the tier of the logic if the player owns the maximum number."""
        total_cells = self.grid_size[0] * self.grid_size[1]
        owned_blocks = self.logic_inventory[logic_id]['count']

        if owned_blocks >= total_cells:
            # Calculate the price per block based on the current tier
            tier = self.logic_inventory[logic_id]['tier']
            price_per_block = self.get_logic_price(logic_id)

            # Initialize the block removal count
            removed_blocks_count = 0

            # Remove all cells of this logic type from the grid
            for row in range(self.grid_size[0]):
                for col in range(self.grid_size[1]):
                    if self.logic_grid[row, col] == logic_id:
                        # Count each block being removed
                        removed_blocks_count += 1
                        # Reset the grid cell to 'Void'
                        self.logic_grid[row, col] = RULESET_IDS["Void"]

            # Ensure we tally the correct number of blocks for the refund
            total_refund = removed_blocks_count * price_per_block

            # Add the refunded amount to the player's bonus (energy)
            self.bonus += total_refund

            # Print the refunded energy to the console
            print(f"Refunded {self.format_number(total_refund)} energy for upgrading {ID_RULESETS[logic_id]} to Tier {tier + 1}.")

            # Upgrade tier and reset count
            self.logic_inventory[logic_id]['count'] = 0  # Reset count for new tier
            self.logic_inventory[logic_id]['tier'] += 1

            # Print the tier upgrade confirmation
            print(f"{ID_RULESETS[logic_id]} upgraded to Tier {self.logic_inventory[logic_id]['tier']}!")

    def reset_game(self):
        """Reset the game grid based on the current mode."""
        if self.mode == 'building':
            # Refund logic blocks before resetting
            unique_logic_ids, counts = np.unique(self.logic_grid, return_counts=True)
            for logic_id, count in zip(unique_logic_ids, counts):
                if logic_id != 0 and logic_id not in self.infinite_logic_ids:
                    self.logic_inventory[logic_id]['count'] += count
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
        self.save_current_state()
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
            #print("Cannot flood-fill with Void logic over Void cells.")
            return

        # If the selected cell is already the same ruleset, return
        if self.logic_grid[row, col] == self.selected_ruleset_id:
            return

        # Set of locations to be filled, starting with the initial location
        locs = set()
        initial_value = self.logic_grid[row, col]

        # The initial cell coordinates are added to the locs set
        locs.add((row, col))

        # Start flood-filling
        while locs:
            # Pop a cell from the set to process
            r, c = locs.pop()

            # Ensure we are within bounds
            if 0 <= r < self.grid_size[0] and 0 <= c < self.grid_size[1]:
                # Only fill cells that have the initial value
                if self.logic_grid[r, c] == initial_value:
                    # Check if player has blocks of the selected logic type or if it's infinite
                    if self.logic_inventory.get(self.selected_ruleset_id, {'count': 0})['count'] > 0 or self.selected_ruleset_id in self.infinite_logic_ids:
                        # Refund the old logic block if it's not Void and not infinite
                        old_logic_id = self.logic_grid[r, c]
                        if old_logic_id != RULESET_IDS["Void"] and old_logic_id not in self.infinite_logic_ids:
                            self.logic_inventory[old_logic_id]['count'] += 1
                            
                        # Apply selected ruleset
                        self.logic_grid[r, c] = self.selected_ruleset_id
                        
                        # Decrease inventory if not infinite
                        if self.selected_ruleset_id not in self.infinite_logic_ids:
                            self.logic_inventory[self.selected_ruleset_id]['count'] -= 1

                        # Add new coordinates to the set based on the fill pattern (similar to mighty_wind)
                        for a in self.fill_patterns[self.current_pattern_index]:
                            new_r, new_c = r + a[0], c + a[1]
                            if 0 <= new_r < self.grid_size[0] and 0 <= new_c < self.grid_size[1]:
                                if self.logic_grid[new_r, new_c] == initial_value:
                                    locs.add((new_r, new_c))
                    else:
                        #print("Not enough blocks of this logic type.")
                        return
                    
    def calculate_physical_defense_and_offense(self):
        """Calculate physical defense based on static living cells, and offense based on active living cells."""
        # Ensure that we have a previous cell state grid to compare with
        if not hasattr(self, 'previous_cell_state_grid'):
            self.previous_cell_state_grid = (self.cell_state_grid != self.dead_cell_index).copy()  # Only track living state, not color
            return 0, 0  # Return 0 for both defense and offense the first time

        # Track the "alive" state of the current and previous grids (ignore color, just check if cells are alive or dead)
        current_alive_state = (self.cell_state_grid != self.dead_cell_index)
        previous_alive_state = (self.previous_cell_state_grid != self.dead_cell_index)

        # Count static living cells (defense) — cells that are alive in both states
        static_cells_count = np.sum(current_alive_state & previous_alive_state)

        # Count total living cells
        total_living_cells = np.sum(current_alive_state)

        # Offense is the number of active (changing) cells, which is total living cells minus static cells
        active_cells_count = total_living_cells - static_cells_count

        # Debugging: Print the number of static and active cells
        #print(f"Static cells (defense): {static_cells_count}, Active cells (offense): {active_cells_count}, Total living cells: {total_living_cells}")

        # Do not update previous_cell_state_grid here; it's updated in the update() method

        return static_cells_count, active_cells_count

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
        """Handle mouse interactions with dynamic brush size based on pause state."""
        pos = pygame.mouse.get_pos()
        self.save_current_state()
        
        col, row = self.get_grid_position(pos)
        if col is None or row is None:
            return  # Ignore clicks outside the grid surface

        brush_offsets = self.get_brush_offsets()

        if self.mode == 'building':
            self.handle_building_mode(row, col, brush_offsets)
        elif self.mode == 'simulation':
            self.handle_simulation_mode(row, col, brush_offsets)

    def get_grid_position(self, pos):
        """Adjusts the mouse position based on fullscreen and returns the grid position (col, row)."""
        if self.fullscreen:
            grid_x_offset = (self.screen.get_width() - self.grid_surface.get_width()) // 2
            grid_y_offset = (self.screen.get_height() - self.grid_surface.get_height()) // 2

            adjusted_x = pos[0] - grid_x_offset
            adjusted_y = pos[1] - grid_y_offset - 30  # Adjust for UI height

            if 0 <= adjusted_x < self.grid_surface.get_width() and 0 <= adjusted_y < self.grid_surface.get_height():
                col = adjusted_x // (self.cell_size + self.margin)
                row = adjusted_y // (self.cell_size + self.margin)
            else:
                return None, None  # Mouse outside grid surface
        else:
            col = pos[0] // (self.cell_size + self.margin)
            row = (pos[1] - 30) // (self.cell_size + self.margin)
        
        return col, row

    def get_brush_offsets(self):
        """Returns the brush offsets based on the current mode and pause state."""
        if self.mode == 'simulation' and not self.paused:
            return [(0, 0), (0, 1), (1, 0), (1, 1)]  # 2x2 brush when unpaused
        return [(0, 0)]  # 1x1 brush when paused or in building mode

    def handle_building_mode(self, row, col, brush_offsets):
        """Handles mouse actions in building mode."""
        for dr, dc in brush_offsets:
            current_row = row + dr
            current_col = col + dc
            if not (0 <= current_row < self.grid_size[0] and 0 <= current_col < self.grid_size[1]):
                continue

            if self.mouse_buttons[0]:  # Left click to assign logic
                self.assign_logic_block(current_row, current_col)

            if self.mouse_buttons[2]:  # Right click to erase logic (set to Void state)
                self.erase_logic_block(current_row, current_col)

    def assign_logic_block(self, row, col):
        """Assigns a logic block to the grid at the specified row and column."""
        if self.logic_inventory.get(self.selected_ruleset_id, {'count': 0})['count'] > 0 or self.selected_ruleset_id in self.infinite_logic_ids:
            old_logic_id = self.logic_grid[row, col]
            if old_logic_id != self.selected_ruleset_id:
                if old_logic_id != RULESET_IDS["Void"] and old_logic_id not in self.infinite_logic_ids:
                    self.logic_inventory[old_logic_id]['count'] += 1  # Refund the old logic block
                self.logic_grid[row, col] = self.selected_ruleset_id
                if self.selected_ruleset_id not in self.infinite_logic_ids:
                    self.logic_inventory[self.selected_ruleset_id]['count'] -= 1

    def erase_logic_block(self, row, col):
        """Erases a logic block from the grid and sets it to Void state."""
        old_logic_id = self.logic_grid[row, col]
        if old_logic_id != RULESET_IDS["Void"]:
            if old_logic_id not in self.infinite_logic_ids:
                self.logic_inventory[old_logic_id]['count'] += 1  # Increment inventory for removed block
            self.logic_grid[row, col] = RULESET_IDS["Void"]

    def handle_simulation_mode(self, row, col, brush_offsets):
        """Handles mouse actions in simulation mode."""
        for dr, dc in brush_offsets:
            current_row = row + dr
            current_col = col + dc
            if not (0 <= current_row < self.grid_size[0] and 0 <= current_col < self.grid_size[1]):
                continue

            if self.mouse_buttons[0]:  # Left click to paint living cells
                self.paint_cell(current_row, current_col)

            if self.mouse_buttons[2]:  # Right click to erase cells
                self.erase_cell(current_row, current_col)

    def paint_cell(self, row, col):
        """Paints a cell in the grid and handles bonuses if the cell was dead."""
        was_dead = self.cell_state_grid[row, col] == self.dead_cell_index
        self.cell_state_grid[row, col] = self.selected_color_index
        if was_dead:
            self.bonus += 1  # Award bonus for painting a dead cell
            self.painted_cells_during_pause.add((row, col))

    def erase_cell(self, row, col):
        """Erases a cell in the grid, setting it back to its dead state."""
        if self.cell_state_grid[row, col] != self.dead_cell_index:
            self.cell_state_grid[row, col] = self.dead_cell_index
            self.painted_cells_during_pause.discard((row, col))

    def calculate_upgrade_cost(self, logic_id):
        """Calculate the cost of upgrading to the next tier for a specific logic type."""
        current_tier = self.logic_inventory[logic_id]['tier']
        price_per_block = self.get_logic_price(logic_id)
        total_cells = self.grid_size[0] * self.grid_size[1]
        placed_blocks = np.sum(self.logic_grid == logic_id)

        # Increase the cost exponentially as the tier increases
        upgrade_cost = placed_blocks * price_per_block * current_tier  # Cost increases with the current tier
        return upgrade_cost

    def purchase_logic_block(self, logic_id, quantity):
        """Attempt to purchase the specified quantity of logic blocks."""
        total_cells = self.grid_size[0] * self.grid_size[1]
        placed_blocks = np.sum(self.logic_grid == logic_id)
        owned_blocks = self.logic_inventory[logic_id]['count']
        max_additional_blocks = total_cells - (placed_blocks + owned_blocks)

        if quantity == '+TIER':
            # Calculate the cost to upgrade the tier
            upgrade_cost = self.calculate_upgrade_cost(logic_id)

            # Check if the player has enough energy for the upgrade
            if self.bonus >= upgrade_cost:
                # Proceed with the upgrade
                self.bonus -= upgrade_cost

                # Remove all blocks of the current tier
                removed_blocks_count = 0
                for row in range(self.grid_size[0]):
                    for col in range(self.grid_size[1]):
                        if self.logic_grid[row, col] == logic_id:
                            removed_blocks_count += 1
                            self.logic_grid[row, col] = RULESET_IDS["Void"]

                # Refund based on the removed blocks
                total_refund = removed_blocks_count * self.get_logic_price(logic_id)
                self.bonus += total_refund

                # Upgrade tier and reset the block count
                self.logic_inventory[logic_id]['count'] = 0
                self.logic_inventory[logic_id]['tier'] += 1

                print(f"Upgraded {ID_RULESETS[logic_id]} to Tier {self.logic_inventory[logic_id]['tier']}.")
                print(f"Refunded {self.format_number(total_refund)} energy for removed blocks.")
            else:
                print(f"Not enough energy to upgrade {ID_RULESETS[logic_id]} to Tier {self.logic_inventory[logic_id]['tier'] + 1}. "
                      f"Required: {self.format_number(upgrade_cost)}, Available: {self.format_number(self.bonus)}")

        else:
            # Handle regular block purchasing
            quantity = min(quantity, max_additional_blocks)
            if quantity <= 0:
                print("You already own or have placed the maximum number of blocks for this grid.")
                return

            price_per_block = self.get_logic_price(logic_id)
            total_price = price_per_block * quantity

            if self.bonus >= total_price:
                self.bonus -= total_price
                self.logic_inventory[logic_id]['count'] += quantity
                print(f"Purchased {quantity} blocks of {ID_RULESETS[logic_id]} for {self.format_number(total_price)} energy.")
            else:
                print(f"Not enough energy to purchase {quantity} blocks of {ID_RULESETS[logic_id]}. Total cost: {self.format_number(total_price)}.")

    def calculate_initial_energy_burst(self):
        """Calculate energy burst for cells painted during pause."""
        surviving_cells = 0
        for row, col in self.painted_cells_during_pause:
            # Check if the painted cell is still alive after unpausing
            if self.cell_state_grid[row, col] != self.dead_cell_index:
                surviving_cells += 1
        
        # Generate energy based on surviving painted cells
        self.bonus += surviving_cells
        #print(f"Energy Burst: {surviving_cells} energy generated from painted cells.")
        
    def update(self):
        """Update the simulation."""
        if self.mode == 'simulation' and not self.paused:
            # Clear the undo history when the simulation is unpaused
            if self.cell_state_grid_history:
                self.cell_state_grid_history.clear()

            # Store previous cell state grid before update
            if hasattr(self, 'previous_cell_state_grid'):
                previous_cell_state_grid = self.cell_state_grid.copy()
            else:
                previous_cell_state_grid = self.cell_state_grid.copy()

            # Convert logic_inventory to a NumPy array
            num_rulesets = len(RULESETS)
            logic_inventory_array = np.zeros((num_rulesets, 2), dtype=np.int32)
            for logic_id in self.logic_inventory:
                logic_inventory_array[logic_id, 0] = self.logic_inventory[logic_id]['count']  # Store count
                logic_inventory_array[logic_id, 1] = self.logic_inventory[logic_id]['tier']   # Store tier

            # Call update_cells with logic_inventory_array
            self.cell_state_grid, births_count, core_energy_generated = update_cells(
                self.cell_state_grid,
                self.logic_grid,
                self.birth_rules_array,
                self.survival_rules_array,
                self.rule_lengths,
                self.dead_cell_index,
                self.neighbor_offsets,
                self.colors_array,
                logic_inventory_array,  # Pass the logic inventory array
                logic_base_energy_array  # Pass the base energy array here
            )

            # Print debug information for grid update
            #print(f"Updated cell grid. Births: {births_count}, Core energy generated: {core_energy_generated}")
            #print(f"Current cell state grid (sample): \n{self.cell_state_grid[:5, :5]}")  # Print a sample of the grid

            # Track whether any Tier 2 or higher blocks are contributing to energy generation
            tier_2_or_higher_energy = any(
                logic_inventory_array[logic_id, 1] >= 2 for logic_id in range(num_rulesets)
            )

            # Add core energy to the bonus
            self.bonus += core_energy_generated

            # Increment frame counter
            self.frame_counter += 1

            # Accumulate time for bonus calculation
            dt = self.clock.get_time() / 1000.0
            self.bonus_timer += dt

            # Check if it's time to calculate bonuses
            if self.bonus_timer >= self.bonus_interval:
                self.calculate_bonuses()
                self.bonus_timer -= self.bonus_interval  # Reset the timer

            # Accumulate core energy for energy generation tracking
            self.core_energy_generated_last += core_energy_generated

            # Tally colors every 'tally_frequency' frames
            if self.frame_counter % self.tally_frequency == 0:
                self.tally_colors()

                # Calculate the bonus energy based on diversity
                bonus_energy = self.core_energy_generated_last * self.diversity_bonus
                self.total_energy_generated = self.core_energy_generated_last + bonus_energy

                # Add total energy generated to the player bonus
                self.bonus += self.total_energy_generated

                # Update the energy generation rate (for display purposes)
                self.energy_generation_timer += self.simulation_interval * self.tally_frequency

                if self.energy_generation_timer >= 1.0:
                    # Calculate energy_generation_rate based on both core and bonus energy
                    self.energy_generation_rate = self.total_energy_generated / self.energy_generation_timer

                    # Reset counters
                    self.core_energy_generated_last = 0
                    self.energy_generation_timer = 0.0

                # Reset energy_generated_last for the next tally
                self.energy_generated_last = 0
                
            # Calculate offense and defense before updating previous_cell_state_grid
            self.defense, self.offense = self.calculate_physical_defense_and_offense()

            # Update the previous cell state grid
            self.previous_cell_state_grid = self.cell_state_grid.copy()
            
    def save_current_state(self):
        """Save the current state of grids and logic inventory for undo."""
        if self.mode == 'building':
            # Only save state for building mode when using the fill bucket
            if len(self.logic_grid_history) == 0 or np.any(self.logic_grid != self.logic_grid_history[-1][0]):
                self.logic_grid_history.append((self.logic_grid.copy(), self.logic_inventory.copy()))
            # Limit history size
            if len(self.logic_grid_history) > self.max_history_length:
                self.logic_grid_history.pop(0)
        elif self.mode == 'simulation' and self.paused:
            # Save state whenever there is a change in the grid
            if len(self.cell_state_grid_history) == 0 or np.any(self.cell_state_grid != self.cell_state_grid_history[-1]):
                self.cell_state_grid_history.append(self.cell_state_grid.copy())
            # Limit history size
            if len(self.cell_state_grid_history) > self.max_history_length:
                self.cell_state_grid_history.pop(0)

    def undo_action(self):
        """Undo the last action in the current mode."""
        if self.mode == 'building' and self.logic_grid_history:
            # Get the previous state from the history stack in building mode
            previous_logic_grid, previous_logic_inventory = self.logic_grid_history.pop()

            # Compare the current grid and the previous grid to find what was placed or changed
            for row in range(self.grid_size[0]):
                for col in range(self.grid_size[1]):
                    current_logic = self.logic_grid[row, col]
                    previous_logic = previous_logic_grid[row, col]

                    if current_logic != previous_logic:
                        # If a block was placed, refund it to the player's inventory
                        if current_logic != RULESET_IDS["Void"] and current_logic not in self.infinite_logic_ids:
                            self.logic_inventory[current_logic]['count'] += 1

                        # If a block was removed, deduct from the player's inventory
                        if previous_logic != RULESET_IDS["Void"] and previous_logic not in self.infinite_logic_ids:
                            self.logic_inventory[previous_logic]['count'] -= 1

            # Restore the previous grid and inventory state
            self.logic_grid = previous_logic_grid
            self.logic_inventory = previous_logic_inventory.copy()

        elif self.mode == 'simulation' and self.paused and self.cell_state_grid_history:
            # Undo action for the paused simulation mode (painting mode)
            previous_cell_state_grid = self.cell_state_grid_history.pop()

            # Compare the current grid and the previous grid to find what was painted or removed
            for row in range(self.grid_size[0]):
                for col in range(self.grid_size[1]):
                    current_cell = self.cell_state_grid[row, col]
                    previous_cell = previous_cell_state_grid[row, col]

                    # If the cell was painted in the current state but not in the previous state
                    if current_cell != previous_cell:
                        # Optionally: Track any specific undo logic related to painted cells
                        if current_cell != self.dead_cell_index:
                            # If a cell was painted, undo the painting
                            self.cell_state_grid[row, col] = previous_cell

                        # If a cell was erased, undo the erasing
                        if current_cell == self.dead_cell_index and previous_cell != self.dead_cell_index:
                            self.cell_state_grid[row, col] = previous_cell

            # Restore the previous cell state grid
            self.cell_state_grid = previous_cell_state_grid

    def calculate_bonuses(self):
        """Calculate bonuses based on cell activity and color diversity."""
        # Calculate Offense and Defense based on new and static cells
        self.bonus_offense = self.minute_new_cells * 1.0  # Each new cell adds +1 attack
        self.bonus_defense = self.minute_static_cells * 1.0  # Each static cell adds +1 defense

        # Calculate color diversity bonus
        total_living_cells = np.sum(self.minute_color_counts)
        if total_living_cells > 0:
            color_proportions = self.minute_color_counts / total_living_cells
            simpson_index = 1.0 / np.sum(color_proportions ** 2)
            max_simpson_index = len(PRIMARY_COLORS)
            self.diversity_bonus = (simpson_index / max_simpson_index) * 0.1  # Diversity bonus as a percentage
        else:
            self.diversity_bonus = 0.0

        # Update the total bonus energy rate
        self.bonus_energy_rate = self.core_energy_generated_last * self.diversity_bonus

        # Reset accumulators
        self.minute_cell_changes = 0
        self.minute_new_cells = 0
        self.minute_static_cells = 0
        self.minute_color_counts[:] = 0

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
                        distance = self.color_distance(cell_color, primary_color)
                        if distance < min_distance:
                            min_distance = distance
                            closest_primary_idx = idx
                    if closest_primary_idx != -1:
                        self.color_counts[closest_primary_idx] += 1
        # Accumulate for bonus calculation
        self.minute_color_counts += self.color_counts

    def draw_menu(self):
        """Draw the menu with the previous mode as the background."""
        self.draw_game()  # Draw the background grid from the previous mode

        self.draw_menu_overlay()  # Semi-transparent overlay
        box_x, box_y, box_width, box_height = self.get_popup_box_dimensions()
        self.draw_popup_background(box_x, box_y, box_width, box_height)

        font = pygame.font.Font(None, 28)
        self.copy_button_rect = self.draw_button('Copy', box_x + box_width // 4, box_y + 60, font)
        self.paste_button_rect = self.draw_button('Paste', box_x + 3 * box_width // 4, box_y + 60, font)

        self.draw_copypaste_instructions(font, box_x, box_y, box_width)

    def draw_menu_overlay(self):
        """Draw the semi-transparent black overlay."""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # Semi-transparent black
        self.screen.blit(overlay, (0, 0))

    def get_popup_box_dimensions(self):
        """Return the dimensions and position of the popup box."""
        box_width = 600
        box_height = 200
        box_x = (self.width - box_width) // 2
        box_y = (self.height - box_height) // 2
        return box_x, box_y, box_width, box_height

    def draw_popup_background(self, box_x, box_y, box_width, box_height):
        """Draw the background of the popup."""
        popup_rect = pygame.Rect(box_x, box_y, box_width, box_height)
        pygame.draw.rect(self.screen, (50, 50, 50), popup_rect)

    def draw_button(self, text, x, y, font):
        """Draw a button with the given text, position, and font. Return the button rect."""
        button_width = 60
        button_height = 30
        button_rect = pygame.Rect(x - button_width // 2, y, button_width, button_height)
        
        pygame.draw.rect(self.screen, (70, 70, 70), button_rect)  # Draw button background
        text_surface = font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=button_rect.center)
        
        self.screen.blit(text_surface, text_rect)  # Draw button text
        return button_rect
    
    def draw_copypaste_instructions(self, font, box_x, box_y, box_width):
        """Draw the instruction text centered below the buttons."""
        instruction_text = 'Use Copy to copy the seed and Paste to load it.'
        instruction_surface = font.render(instruction_text, True, (255, 255, 255))
        instruction_rect = instruction_surface.get_rect(center=(self.width // 2, box_y + 120))
        self.screen.blit(instruction_surface, instruction_rect)

    def get_logic_price(self, logic_id):
        """Get the price per block for the logic, considering its tier."""
        base_price = self.logic_prices[logic_id]  # This should be the base price for the logic type
        tier = self.logic_inventory[logic_id]['tier']  # Retrieve the current tier for this logic
        
        # If higher tiers increase the price, we scale by a factor (e.g., doubling per tier)
        price_per_block = base_price * (10 ** (tier - 1))  # Exponential price increase per tier
        
        return price_per_block

    def draw_shop(self):
        """Draw the buy menu with the current mode (building or simulation) as the background."""
        self.draw_game()  # Draw the background grid from the current mode
        font = pygame.font.Font(None, 28)

        available_logic_ids = self.get_available_logic_ids()
        max_logic_name_width, max_price_width = self.calculate_max_text_widths(font, available_logic_ids)

        box_width, box_height, box_x, box_y = self.calculate_popup_dimensions(
            max_logic_name_width, max_price_width, len(available_logic_ids)
        )
        self.draw_popup_background(box_x, box_y, box_width, box_height)

        self.draw_balance_text(font, box_x, box_y)
        self.draw_logic_items(font, available_logic_ids, max_logic_name_width, max_price_width, box_x, box_y)

    def get_available_logic_ids(self):
        """Return the list of logic IDs that are available for purchase."""
        return [logic_id for logic_id in self.logic_prices.keys() 
                if logic_id not in (RULESET_IDS["Conway"], RULESET_IDS["Void"])]

    def calculate_max_text_widths(self, font, available_logic_ids):
        """Calculate the maximum text width for logic names and prices."""
        max_logic_name_width = 0
        max_price_width = 0

        for logic_id in available_logic_ids:
            logic_name, price_text = self.get_logic_name_and_price_text(logic_id)
            max_logic_name_width = max(max_logic_name_width, font.size(logic_name)[0])
            max_price_width = max(max_price_width, font.size(price_text)[0])

        return max_logic_name_width, max_price_width

    def calculate_popup_dimensions(self, max_logic_name_width, max_price_width, logic_count):
        """Calculate and return the popup box dimensions."""
        button_area_width = (70 + 10) * 3  # Width for 3 buttons with spacing
        box_width = max_logic_name_width + max_price_width + button_area_width + 80
        box_height = 80 + logic_count * 50
        box_x = (self.width - box_width) // 2
        box_y = (self.height - box_height) // 2
        return box_width, box_height, box_x, box_y

    def draw_popup_background(self, box_x, box_y, box_width, box_height):
        """Draw the popup background."""
        popup_rect = pygame.Rect(box_x, box_y, box_width, box_height)
        pygame.draw.rect(self.screen, (50, 50, 50), popup_rect)

    def draw_balance_text(self, font, box_x, box_y):
        """Draw the player's current balance at the top of the popup."""
        balance_text = font.render(f"Current Energy: {self.format_number(self.bonus)}", True, (255, 255, 255))
        balance_rect = balance_text.get_rect(center=(self.width // 2, box_y + 20))
        self.screen.blit(balance_text, balance_rect)

    def get_logic_name_and_price_text(self, logic_id):
        """Return the logic name (with tier) and price text for the given logic_id."""
        logic_name = ID_RULESETS[logic_id]
        tier = self.logic_inventory[logic_id]['tier']
        price_per_block = self.get_logic_price(logic_id)
        
        logic_text = f"{logic_name}" if tier == 1 else f"{tier} {logic_name}"
        price_text = f"Price per block: {self.format_number(price_per_block)}"
        return logic_text, price_text

    def draw_logic_items(self, font, available_logic_ids, max_logic_name_width, max_price_width, box_x, box_y):
        """Draw the logic names, prices, and buy buttons."""
        y_offset = box_y + 60
        self.buy_buttons = []  # Reset the buy buttons list

        for logic_id in available_logic_ids:
            logic_name, price_text = self.get_logic_name_and_price_text(logic_id)

            self.draw_text(font, logic_name, (box_x + 20, y_offset))  # Draw logic name
            self.draw_text(font, price_text, (box_x + max_logic_name_width + 40, y_offset))  # Draw price
            
            self.draw_buy_buttons(font, logic_id, max_logic_name_width, max_price_width, box_x, y_offset)
            y_offset += 50  # Adjust for next item

    def draw_text(self, font, text, position):
        """Helper function to draw text on the screen."""
        surface = font.render(text, True, (255, 255, 255))
        self.screen.blit(surface, position)

    def draw_buy_buttons(self, font, logic_id, max_logic_name_width, max_price_width, box_x, y_offset):
        """Draw the buy buttons (Buy: 1 | 25 | +TIER) for a logic type."""
        start_x = box_x + max_logic_name_width + max_price_width + 80
        self.draw_text(font, "Buy:", (start_x, y_offset))  # Draw "Buy:" label

        start_x += font.size("Buy: ")[0] + 10  # Adjust for buttons

        quantities = [1, 25, '+TIER']
        for qty in quantities:
            button_text = str(qty)
            button_width = font.size(button_text)[0] + 20

            button_rect = pygame.Rect(start_x, y_offset - 5, button_width, 30)
            pygame.draw.rect(self.screen, (70, 70, 70), button_rect)

            self.draw_text_centered(font, button_text, button_rect)
            self.buy_buttons.append((button_rect, logic_id, qty))

            start_x += button_width + 10  # Move to next button

    def draw_text_centered(self, font, text, rect):
        """Draw text centered within a given rectangle."""
        text_surface = font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

    def format_number(self, number):
        """Format the number based on the user's notation preference (scientific or normal)."""
        if self.scientific_notation:
            # Format the number in scientific notation if the user prefers it
            return f"{number:.3e}"  # You can adjust the precision (.3e means 3 decimal places)
        else:
            # Format the number normally with commas for thousands
            return f"{number:,.0f}"  # Use commas and no decimal places

    def draw(self):
        """Draw the grid and UI elements."""
        self.screen.fill((90, 90, 180))  # Clear screen

        if self.mode == 'menu':
            # Draw the game in the background
            self.draw_game()
            self.draw_menu()
        elif self.mode == 'shop':
            self.draw_game()
            self.draw_shop()
        elif self.mode == 'stats':
            # Draw stats popup over the current game screen
            self.draw_game()
            self.draw_stats()
        else:
            # Draw the grid in either building or simulation mode
            self.draw_game()

    def draw_game(self):
        """Draw the game grid and UI based on the current mode."""
        
        mode_to_draw = self.get_mode_to_draw()
        
        # Draw the grid
        self.draw_grid(mode_to_draw)
        
        # Draw the appropriate UI based on the mode
        if self.mode == 'simulation':
            self.draw_simulation_ui()
        else:
            self.draw_building_ui()
        
        # Draw instructions at the bottom of the screen
        self.draw_instructions()

    def get_mode_to_draw(self):
        """Determine which mode to use for drawing."""
        if self.mode in ['building', 'simulation']:
            return self.mode
        elif self.mode in ['shop', 'menu', 'stats']:
            return self.previous_mode
        return self.mode

    def draw_grid(self, mode_to_draw):
        """Draw the game grid based on the mode."""
        if self.fullscreen:
            self.grid_surface.fill((0, 0, 0))
            surface = self.grid_surface
        else:
            surface = self.screen
        
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                x, y = self.calculate_grid_cell_position(col, row)
                color = self.get_cell_color(mode_to_draw, row, col)
                pygame.draw.rect(surface, color, (x, y, self.cell_size, self.cell_size))
        
        if self.fullscreen:
            grid_x = (self.screen.get_width() - self.grid_surface.get_width()) // 2
            grid_y = (self.screen.get_height() - self.grid_surface.get_height()) // 2
            self.screen.blit(self.grid_surface, (grid_x, grid_y))

    def calculate_grid_cell_position(self, col, row):
        """Calculate the pixel position of the grid cell for drawing."""
        x = col * (self.cell_size + self.margin) + self.margin
        y = row * (self.cell_size + self.margin) + self.margin + 30  # Adjust for UI height
        return x, y

    def get_cell_color(self, mode_to_draw, row, col):
        """Determine the color of the cell based on the mode."""
        if mode_to_draw == 'building':
            ruleset_id = self.logic_grid[row, col]
            return self.logic_colors_list.get(ruleset_id, (50, 50, 50))
        else:  # Simulation mode
            color_index = self.cell_state_grid[row, col]
            return self.colors_array[color_index]

    def draw_simulation_ui(self):
        """Draw the UI specific to simulation mode."""
        font = pygame.font.SysFont(None, 32)
        font_large = pygame.font.SysFont(None, 42)
        
        mode_text_surface = self.render_text_with_outline("Mode: Simulation", font, (255, 255, 255), (0, 0, 0))
        self.screen.blit(mode_text_surface, (10, 5))

        total_living_cells = np.sum(self.color_counts)
        self.draw_alive_text(total_living_cells, font)
        
        if self.paused:
            self.draw_paused_text(font_large)
        else:
            self.draw_energy_text(font)
        
    def draw_alive_text(self, total_living_cells, font):
        """Draw the alive cell count and ratio."""
        alive_text = f"Alive: {total_living_cells} | Ratio:"
        alive_text_surface = self.render_text_with_outline(alive_text, font, (255, 255, 255), (0, 0, 0))
        alive_text_rect = alive_text_surface.get_rect()
        
        ratio_surfaces, ratio_text_width = self.get_ratio_surfaces(total_living_cells, font)
        
        total_text_width = alive_text_rect.width + ratio_text_width
        alive_text_x_position = self.width - total_text_width - 10
        self.screen.blit(alive_text_surface, (alive_text_x_position, 5))
        
        ratio_text_x_position = alive_text_x_position + alive_text_rect.width + 10
        for ratio_surface in ratio_surfaces:
            self.screen.blit(ratio_surface, (ratio_text_x_position, 5))
            ratio_text_x_position += ratio_surface.get_width() + 10

    def get_ratio_surfaces(self, total_living_cells, font):
        """Generate surfaces for the ratio text of different colors."""
        ratio_surfaces = []
        ratio_text_width = 0
        for idx, count in enumerate(self.color_counts):
            percentage = (count / total_living_cells * 100) if total_living_cells > 0 else 0
            ratio_text = f"{int(percentage)}"
            ratio_surface = self.render_text_with_outline(ratio_text, font, PRIMARY_COLORS[idx], (0, 0, 0))
            ratio_surfaces.append(ratio_surface)
            ratio_text_width += ratio_surface.get_width() + 10
        return ratio_surfaces, ratio_text_width

    def draw_paused_text(self, font_large):
        """Draw the paused text with colorful letters."""
        text = "PAUSED - Painting Time!"
        text_surfaces = [self.render_text_with_outline(char, font_large, PRIMARY_COLORS[i % len(PRIMARY_COLORS)], (127, 127, 127)) 
                         for i, char in enumerate(text)]
        total_width = sum(surface.get_width() for surface in text_surfaces)
        x = (self.width - total_width) // 2
        y = 5
        for surface in text_surfaces:
            self.screen.blit(surface, (x, y))
            x += surface.get_width()

    def draw_energy_text(self, font):
        """Draw the energy and alive cell counts."""
        bonus_text = f"Energy: {self.format_number(self.bonus)} (+{self.format_number(self.energy_generation_rate + self.bonus_energy_rate)}/s)"
        bonus_surface = self.render_text_with_outline(bonus_text, font, (255, 255, 255), (0, 0, 0))
        x = (self.width - bonus_surface.get_width()) // 2
        self.screen.blit(bonus_surface, (x, 1))

    def draw_building_ui(self):
        """Draw the UI specific to building mode."""
        font = pygame.font.SysFont(None, 32)
        
        mode_text_surface = self.render_text_with_outline(f"Mode: {self.mode.capitalize()}", font, (255, 255, 255), (0, 0, 0))
        self.screen.blit(mode_text_surface, (10, 5))
        
        ruleset_text_surface = self.render_text_with_outline(f"Selected Ruleset: {self.selected_ruleset_name}", font, (255, 255, 255), (0, 0, 0))
        self.screen.blit(ruleset_text_surface, (200, 5))
        
        self.draw_logic_inventory()

    def draw_logic_inventory(self):
        """Draw the logic inventory in building mode."""
        stats_font = pygame.font.SysFont(None, 28)
        x_position = self.width - 10
        logic_ids_order = sorted([ruleset_id for ruleset_name, ruleset_id in RULESET_IDS.items() if ruleset_name not in ("Conway", "Void")])

        for logic_id in reversed(logic_ids_order):
            logic_name = ID_RULESETS[logic_id]
            count_text = f"{self.logic_inventory[logic_id]['count']}"
            logic_color = self.logic_colors_list.get(logic_id, (255, 255, 255))
            text_surface = self.render_text_with_outline(f"{logic_name}: {count_text}", stats_font, logic_color, None)
            x_position -= text_surface.get_width()
            self.screen.blit(text_surface, (x_position, 5))
            x_position -= 10

    def draw_instructions(self):
        """Draw instructions at the bottom of the screen."""
        font = pygame.font.SysFont(None, 32)
        instructions = "Tab: Switch Mode | A and D: Change Ruleset/Color | B: Buy | R: Reset | F: Fill Bucket | N: Notation | I: Info (Stats)"
        instructions_surface = self.render_text_with_outline(instructions, font, (255, 255, 255), (0, 0, 0))
        self.screen.blit(instructions_surface, (10, self.height - 25))

    def handle_save(self):
        # Generate the key with energy included
        key = self.serialize_state(self.logic_grid, self.cell_state_grid, self.logic_inventory, self.bonus)

        # Copy the key to the clipboard (as you already handle this)
        if self.clipboard_available:
            pyperclip.copy(key)
            print("Seed copied to clipboard.")

    def handle_load(self):
        # Paste the key from the clipboard
        if self.clipboard_available:
            try:
                clipboard_content = pyperclip.paste()
                try:
                    # Deserialize the state (including energy)
                    self.logic_grid, self.cell_state_grid, self.logic_inventory, self.bonus = self.deserialize_state(clipboard_content, self.grid_size)
                    print("Seed loaded from clipboard.")
                except Exception as e:
                    print(f"Error loading seed: {e}")
            except Exception as e:
                print(f"Error accessing clipboard: {e}")
        else:
            print("Clipboard not available.")
            
    def generate_state_from_seed(self, seed_str):
        """Generate game state from any string seed."""
        if seed_str.lower() == 'godmode':
            # Activate godmode
            total_cells = self.grid_size[0] * self.grid_size[1]
            self.logic_inventory = {
                ruleset_id: {'count': total_cells, 'tier': 1} for ruleset_id in RULESET_IDS.values()
            }
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
            # Generate random logic_inventory with tiers
            self.logic_inventory = {
                ruleset_id: {
                    'count': np.random.randint(0, self.grid_size[0] * self.grid_size[1]),
                    'tier': 1  # Initialize all tiers to 1
                }
                for ruleset_id in RULESET_IDS.values()
            }
            print("Generated game state from seed.")

    def draw_stats(self):
        """Draw the stats popup over the current game screen."""
        self.draw_game()  # Draw the background grid from the previous mode
        font_title = pygame.font.Font(None, 36)
        font = pygame.font.Font(None, 28)

        defense, offense = self.get_defense_and_offense()

        stat_items = self.get_stat_items(defense, offense)
        max_text_width = self.calculate_max_text_width(font, stat_items)

        box_width, box_height, box_x, box_y = self.calculate_stats_popup_dimensions(max_text_width)
        self.draw_popup_background(box_x, box_y, box_width, box_height)

        self.draw_stat_text(font, stat_items, box_x, box_y)
        self.draw_elemental_affinities(font, font_title, box_x, box_y)

        self.draw_close_instructions(font, box_x, box_y, box_height)

    def get_defense_and_offense(self):
        """Return the defense and offense values, defaulting to 0 if not set."""
        defense = getattr(self, 'defense', 0)
        offense = getattr(self, 'offense', 0)
        return defense, offense

    def get_stat_items(self, defense, offense):
        """Return a list of formatted stat items."""
        return [
            f"Energy: {self.format_number(self.bonus)}",
            f"Energy Per Second: {self.format_number(self.energy_generation_rate + self.bonus_energy_rate)}",
            f"Diversity Bonus: {self.diversity_bonus * 100:.2f}%",
            f"Physical Defense: {self.format_number(defense)}",
            f"Physical Offense: {self.format_number(offense)}"
        ]

    def calculate_max_text_width(self, font, stat_items):
        """Calculate and return the maximum width for the stat items text."""
        return max([font.size(item)[0] for item in stat_items])

    def calculate_stats_popup_dimensions(self, max_text_width):
        """Calculate and return the dimensions and position of the stats popup."""
        box_width = max_text_width + 200  # Extra padding
        box_height = 300 + (len(ELEMENTAL_NAMES) * 25)  # Adjust height for elemental affinities
        box_x = (self.width - box_width) // 2
        box_y = (self.height - box_height) // 2
        return box_width, box_height, box_x, box_y

    def draw_popup_background(self, box_x, box_y, box_width, box_height):
        """Draw the background of the stats popup."""
        popup_rect = pygame.Rect(box_x, box_y, box_width, box_height)
        pygame.draw.rect(self.screen, (50, 50, 50), popup_rect)

    def draw_stat_text(self, font, stat_items, box_x, box_y):
        """Draw the stats text inside the popup."""
        y_offset = box_y + 20
        for stat_item in stat_items:
            text_surface = self.render_text_with_outline(stat_item, font, (255, 255, 255), (0, 0, 0))
            self.screen.blit(text_surface, (box_x + 20, y_offset))
            y_offset += 40  # Adjust spacing between stat items

    def draw_elemental_affinities(self, font, font_title, box_x, box_y):
        """Draw the elemental affinities section inside the popup."""
        y_offset = box_y + 20 + len(self.get_stat_items(0, 0)) * 40 + 10  # Calculate y_offset after stats

        elemental_title = "Elemental Affinities:"
        title_surface = self.render_text_with_outline(elemental_title, font_title, (255, 255, 255), (0, 0, 0))
        self.screen.blit(title_surface, (box_x + 20, y_offset))
        y_offset += 30  # Space between title and elemental affinities

        self.tally_colors()
        total_living_cells = np.sum(self.color_counts)

        for idx, count in enumerate(self.color_counts):
            element_name = ELEMENTAL_NAMES[idx]
            proportion = (count / total_living_cells * 100) if total_living_cells > 0 else 0.0
            element_text = f"{element_name}: {proportion:.1f}%"
            element_color = PRIMARY_COLORS[idx]
            element_surface = self.render_text_with_outline(element_text, font, element_color)
            self.screen.blit(element_surface, (box_x + 20, y_offset))
            y_offset += 25  # Space between elemental affinities

    def draw_close_instructions(self, font, box_x, box_y, box_height):
        """Draw the instructions to close the stats popup."""
        instructions = "Press 'I' to close stats."
        instructions_surface = font.render(instructions, True, (255, 255, 255))
        instructions_rect = instructions_surface.get_rect(center=(self.width // 2, box_y + int(box_height * .96)))
        self.screen.blit(instructions_surface, instructions_rect)

    def print_living_cells(self):
        total_living_cells = np.sum(self.cell_state_grid != self.dead_cell_index)
        print(f"Total living cells: {total_living_cells}")

    def render_text_with_outline(self, text, font, text_color, outline_color=None):
        """
        Render text with an outline. If outline_color is not provided, it will calculate a contrasting color.
        """
        # Calculate a contrasting outline color if not provided
        if outline_color is None:
            # Convert the text color to grayscale to determine brightness
            brightness = 0.299 * text_color[0] + 0.587 * text_color[1] + 0.114 * text_color[2]
            outline_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        
        # First, render the text normally
        text_surface = font.render(text, True, text_color)
        
        # Then, create a new surface slightly larger to accommodate the outline
        size = text_surface.get_width() + 2, text_surface.get_height() + 2
        outline_surface = pygame.Surface(size, pygame.SRCALPHA)
        
        # Draw the outline by rendering the text multiple times with small offsets
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in offsets:
            pos = (1 + dx, 1 + dy)
            outline_surface.blit(font.render(text, True, outline_color), pos)
        
        # Blit the main text onto the outline surface
        outline_surface.blit(text_surface, (1, 1))
        
        return outline_surface
    
    def color_distance(self, c1, c2):
        """Calculate Euclidean distance between two RGB colors."""
        return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

    def restructure_colors(self, colors_list):
        # Separate black, white, and dead cell colors
        special_colors = [
            colors_list[-3],  # Black
            colors_list[-2],  # White
            colors_list[-1]   # Dead cell color
        ]
        
        # Restructure only the colors excluding black, white, and dead cell
        sorted_colors = sorted(
            colors_list[:-3],  # Exclude the last three special colors
            key=lambda color: min(self.color_distance(color, primary) for primary in PRIMARY_COLORS)
        )
        
        # Append the special colors (black, white, dead cell) at the end
        return sorted_colors + special_colors

@njit
def shift_grid(grid, shift_row, shift_col):
    """Shift the grid manually along both axes with toroidal wrapping."""
    rows, cols = grid.shape
    result = np.zeros_like(grid)

    for i in range(rows):
        for j in range(cols):
            new_i = (i + shift_row) % rows
            new_j = (j + shift_col) % cols
            result[new_i, new_j] = grid[i, j]
    
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
def update_cells(
    cell_state_grid,
    logic_grid,
    birth_rules_array,
    survival_rules_array,
    rule_lengths,
    dead_cell_index,
    neighbor_offsets,
    colors_array,
    logic_inventory_array,
    logic_base_energy_array  # Pass base energy array as a parameter
):
    rows, cols = cell_state_grid.shape
    new_grid = cell_state_grid.copy()
    num_colors = colors_array.shape[0]
    births_count = 0  # New variable to count births
    energy_generated = 0  # Track energy generated from births

    # 1. Precompute neighbor counts using manual shifting
    live_neighbor_count = convolve(cell_state_grid != dead_cell_index, neighbor_offsets)

    # 2. Precompute color sums for living neighbors
    neighbor_color_sum = compute_color_sum(cell_state_grid, colors_array, neighbor_offsets, dead_cell_index)

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
            is_alive = cell_state_grid[i, j] != dead_cell_index

            if is_alive:
                # Check survival condition
                survived = False
                for k in range(S_len):
                    if live_neighbors == survival_rules[k]:
                        survived = True
                        break
                if not survived:
                    new_grid[i, j] = dead_cell_index
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

                    # Retrieve the tier of the logic block from logic_inventory_array
                    logic_tier = logic_inventory_array[ruleset_id, 1]  # Column 1 stores tier info

                    # Calculate energy to add
                    base_energy = logic_base_energy_array[ruleset_id]  # Use the NumPy array
                    tier_bonus = (logic_tier - 1)  # Each tier beyond 1 adds 1 extra energy

                    # Ensure energy scales correctly with tier
                    energy_to_add = base_energy + tier_bonus

                    # Update total energy
                    energy_generated += energy_to_add

    return new_grid, births_count, energy_generated

@njit
def compute_color_sum(grid, colors_array, offsets, dead_cell_index):
    """Accumulate the color values for each cell's neighbors."""
    rows, cols = grid.shape
    color_sum = np.zeros((rows, cols, 3), dtype=np.float64)  # RGB sum for each cell

    for offset in offsets:
        shifted_grid = shift_grid(grid, offset[0], offset[1])
        for i in range(rows):
            for j in range(cols):
                if shifted_grid[i, j] != dead_cell_index:
                    color_sum[i, j] += colors_array[shifted_grid[i, j]]  # Accumulate color

    return color_sum

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
    app = Factory()
    app.run()
