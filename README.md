---

# Cellular Automata Factory View
![Game Screenshot](https://raw.githubusercontent.com/deviousname/Game-of-Life-Factory/main/ss1.png)


## Overview

This game is a cellular automaton simulation where you build and manipulate grids of cells governed by different rulesets. Players can switch between **building** and **simulation** modes, modify rulesets, manage a logic block inventory, generate energy, and unlock upgrades. Cell diversity and elemental affinities unlock powerful bonuses, making strategic design crucial.

## Key Features
- **Modes**: Toggle between building and simulation modes with `TAB`. Design in **building mode**, and see cells come to life in **simulation mode**.
- **Rulesets**: Multiple rulesets like **Conway's Life**, **HighLife**, and **Replicator** define cell behavior. Customize and upgrade rulesets for stronger effects.
- **Elemental Affinities**: Cells have elemental colors (Fire, Water, Bio, etc.), and their balance grants bonuses. Track **offense** (new cells) and **defense** (static cells) for strategic gains.
- **Energy System**: Cells generate energy based on their activity, which you can use to buy new blocks or upgrade existing ones.
- **Diversity Bonus**: Having a diverse array of cells grants bonus energy. A well-balanced grid is key to maximizing your efficiency.

## Controls

- **`TAB`**: Switch between building and simulation modes.
- **`A`/`D`**: Cycle through rulesets (building) or colors (simulation).
- **`B`**: Open the shop to buy/upgrade logic blocks with energy.
- **`SPACE`**: Pause/unpause the simulation.
- **`F`**: Flood-fill tool for rapid cell placement.
- **`N`**: Toggle scientific notation for large numbers.

## Rulesets

Rulesets control how cells live, die, or reproduce based on their neighbors:

- **Conway**: Classic life rules (survive with 2-3 neighbors, born with 3).
- **HighLife**: Cells are born with 3 or 6 neighbors.
- **Day & Night**: Complex survival and birth conditions.
- **Void**: Clears a space of all cells.
- Other unique rulesets include **Seeds**, **Life Without Death**, **Maze**, **Gnarl**, and **Replicator**.

## Element Affinities & Bonuses

Cells are linked to elemental colors (e.g., Fire, Water, Ice). Track these affinities to gain energy bonuses. Diverse or balanced cell arrangements grant extra energy through a **diversity bonus**.

- **Offense**: Active cell creation boosts offense.
- **Defense**: Static living cells boost defense.
- **Elemental Affinity**: Each color corresponds to an element. Maintaining a mix of elemental cells increases your bonus.

## Shop & Upgrades

- Use **energy** generated from living cells to buy new logic blocks or upgrade existing ones.
- Higher-tier blocks have stronger effects and cost more energy.
- Some blocks (like Conway and Void) are infinite, while others need to be purchased.

## Energy Generation

Your grid's cells continuously generate energy in simulation mode. This energy can be spent in the shop to expand or upgrade your inventory.

- **Diversity Bonus**: Maintaining color diversity increases your energy output.
- **Scientific Notation**: Toggle large-number display using `N`.

## Clipboard Functionality

You can copy and paste game states via the clipboard, allowing you to save your progress or share designs with others.

---
