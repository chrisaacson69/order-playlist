# CLAUDE.md

**Vault:** `C:\Users\Chris.Isaacson\Vault\projects\order-playlist\README.md`

## Overview

DJ playlist ordering tool. Takes a CSV/TSV of tracks with Camelot key, BPM, and
energy values, then reorders them for smooth harmonic mixing using a greedy
nearest-neighbor algorithm. Optimizes for key compatibility, BPM smoothness,
and energy arc (build-up or cooldown).

## Tech Stack

- **Python** (venv at `env/`)
- **pandas** -- CSV/TSV loading, data manipulation
- **argparse** -- CLI interface
- Visual Studio solution file (`order_playlist.sln`)

## Project Structure

```
order_playlist.py    # Entire tool -- Camelot math, scoring, greedy ordering, CLI
env/                 # Local Python venv (pandas installed here)
README.md
```

## How to Run

```bash
# Activate venv
env\Scripts\activate

# Basic usage (auto-detect separator, build mode)
python order_playlist.py tracks.csv

# TSV input, cooldown mode, custom weights
python order_playlist.py tracks.tsv --sep '\t' --mode cooldown --w_key 2.0 --w_bpm 0.15 --w_energy 0.6

# Pick a specific start track, save ordered CSV
python order_playlist.py tracks.csv --start "Martin Roth|Deep Style" --out ordered.csv
```

## Input CSV Format

Required columns: `Artist`, `Title`, `Camelot`, `BPM`, `Energy`
Optional columns: `Danceability`, `Popularity`, `Happiness`, `Duration` (mm:ss)

## Algorithm Details

- **Camelot cost function** scores key transitions 0.0-3.0:
  - 0.0 = same key, 0.5 = relative major/minor, 1.0 = adjacent same-mode,
    1.5 = adjacent cross-mode, 2.0 = energy boost (+2), 3.0 = other
- **Transition cost** = `w_key * key_cost + w_bpm * |bpm_diff| + w_energy * arc_penalty`
- **Greedy nearest-neighbor**: picks lowest-cost next track at each step
- **Tiebreakers**: higher danceability > popularity > happiness
- **Start track**: auto-selects lowest BPM+energy (build mode) or user-specified via regex

## Key Notes

- `--mode build` penalizes energy drops (gradual rise); `cooldown` penalizes rises
- Default weights: w_key=2.0, w_bpm=0.15, w_energy=0.6
- `--start` uses regex matching against "Artist|Title" (case-insensitive)
- Duration parsing handles mm:ss and hh:mm:ss formats
- Camelot codes must be 1-12 followed by A or B (e.g. `10A`, `3B`)
- Output includes total playlist duration if Duration column present
