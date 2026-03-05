# Grid Fungi Ecosystem Simulation

A grid-based evolutionary ecosystem simulator with replay, ecological metrics, and analysis export.

The main simulation code is in:
- `ecosystem_sim/grid fungi simulation.py`

## What This Project Does

This project simulates organisms on a 2D tile world (`50x50` by default). Each organism has inheritable traits (photosynthesis, defense, invasiveness, parasitism, spread threshold, etc.), consumes energy, competes for space, and mutates over generations.

The simulation can:
- run headless for a fixed number of ticks,
- save runs to disk,
- replay runs with visualization,
- export per-tick profiling/analysis files for external AI/statistical review.

## Core Concepts

### Organism Representation

Organisms are stored in parallel arrays (for performance), including:
- `Energy`, `Age`, `tile_of`, `Alive`
- `photosynthetic_ratio`
- `home_court_potency` (defense)
- `invasiveness`
- `parasitism_rate`
- `spread_threshold`
- `mutation_rate`

A tile index maps into:
- `World[tile] -> organism index`
- `Energy_map[tile] -> environmental energy`

### Per-Tick Dynamics

Each tick generally does:
1. Organism energy update (photosynthesis gain minus BMR)
2. Overflow handling to nearby tiles
3. Parasitism drains from adjacent organisms
4. Spread attempts (reproduction/expansion)
5. Combat resolution for contested tiles
6. Environmental energy regeneration
7. Statistics collection

### Evolution and Inheritance

- Offspring inherit parent traits with optional mutation.
- Species IDs are reused for small mutations and split into new species when mutation magnitude passes a threshold.
- Colors track lineage through inherited hue.

## Environment Presets

Choose a preset by editing:
- `ENVIRONMENT_PRESET = "..."`

Available presets include:
- `baseline`
- `symbiosis_soft_competition`
- `symbiosis_patchy`
- `symbiosis_tense`
- `earth_like`
- `mars_harsh`
- `deep_sea`
- `proxima_b`

These are stylized ecological analog presets, not strict physics models.

## Metrics and Observations

The simulation tracks (per tick) metrics such as:
- species richness
- Shannon diversity and evenness
- age structure distribution
- energy Gini coefficient
- territory stability
- trait investment
- trait correlation matrix
- attacker win rate
- parasitism/photosynthesis ratio

## Replay, Save, and Analysis Export

Generated output folders:
- `ecosystem_sim/saved_runs/`
- `ecosystem_sim/analysis_exports/`

Run artifacts can include:
- saved replay data (`.pkl`)
- analysis summary (`*_analysis.json`)
- tick-by-tick profile (`*_tick_profile.jsonl`)
- optional code snapshot (`*_code_snapshot.py`)

## Visualization

The UI displays:
- organism grid
- energy map
- ecological guild map
- time-series charts
- trait correlation matrix heatmap
- records + ecosystem health text panel

Keyboard controls (replay window):
- `S`: save current run
- `O`: load latest run from disk
- `R`: replay latest archived run
- `L`: stop replay
- `Space`: pause/resume replay
- `Left/Right`: step frames

## Performance Notes

Current performance optimizations include:
- cached tile neighbors (`neighbor_cache`)
- spatial occupancy hash for tile lookups (`spatial_occupancy`)
- alive index set (`alive_indices`) to avoid full-array scans
- reduced capture frequency via `FRAME_CAPTURE_STRIDE`
- optional profiling toggle (`ENABLE_AI_PROFILING`)

For fastest simulation runs:
- keep `ENABLE_AI_PROFILING = False`
- increase `FRAME_CAPTURE_STRIDE`

## Requirements

- Python 3.10+
- `numpy`
- `matplotlib`
- `tkinter` (usually bundled with standard Python on Windows)

Install dependencies:

```bash
pip install numpy matplotlib
```

## How To Run

From repository root:

```bash
python "ecosystem_sim/grid fungi simulation.py"
```

Startup mode is controlled in code:
- `STARTUP_MODE = "simulate"` to run new simulation then replay
- `STARTUP_MODE = "replay_latest"` to load and replay the latest saved run

## Project Layout (Relevant Files)

- `ecosystem_sim/grid fungi simulation.py` - main simulation + visualization + export
- `ecosystem_sim/saved_runs/` - persisted replay/run files
- `ecosystem_sim/analysis_exports/` - analysis JSON/JSONL/snapshots

## Notes

This project is designed for ecosystem behavior exploration and rapid iteration, not as a validated biological or planetary physics model.
