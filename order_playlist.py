#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Order a playlist using Camelot harmonic mixing + BPM smoothness + energy arc.

Usage examples:
  python order_playlist.py tracks.csv
  python order_playlist.py tracks.tsv --sep '\t' --mode build --w_key 2.0 --w_bpm 0.15 --w_energy 0.6
  python order_playlist.py tracks.csv --start "Martin Roth|Deep Style" --out ordered.csv
"""
import argparse
import math
import re
from typing import Tuple, Optional, List

import pandas as pd

# ---------------------------
# Parsing & feature helpers
# ---------------------------

_CAM_REGEX = re.compile(r"^\s*(\d{1,2})\s*([ABab])\s*$")

def parse_camelot(code: str) -> Tuple[int, str]:
    """
    Parse '10A' -> (10, 'A'). Returns (number, mode).
    Raises ValueError for invalid values.
    """
    if not isinstance(code, str):
        raise ValueError(f"Invalid Camelot code {code!r}")
    m = _CAM_REGEX.match(code.strip())
    if not m:
        raise ValueError(f"Invalid Camelot code {code!r}")
    n = int(m.group(1))
    if n < 1 or n > 12:
        raise ValueError(f"Camelot number out of range: {n}")
    mode = m.group(2).upper()
    return n, mode


def circular_distance(a: int, b: int, modulo: int = 12) -> int:
    """Minimal steps on a circle between a and b (1..12 inclusive)."""
    # Shift to 0..11
    a0 = (a - 1) % modulo
    b0 = (b - 1) % modulo
    d = abs(a0 - b0)
    return min(d, modulo - d)


def camelot_cost(a_code: str, b_code: str) -> float:
    """
    Lower is better. Implements common harmonic-mixing preferences:
      0.0: identical key (e.g., 10A -> 10A)
      0.5: relative major/minor (10A <-> 10B)
      1.0: adjacent same-mode (10A -> 9A or 11A)
      1.5: adjacent cross-mode (10A -> 9B or 11B)
      2.0: energy-boost (+2 same mode) (10A -> 12A or 8A)
      3.0: other transitions (discouraged but allowed)
    """
    an, am = parse_camelot(a_code)
    bn, bm = parse_camelot(b_code)

    if an == bn and am == bm:
        return 0.0
    if an == bn and am != bm:
        return 0.5

    d = circular_distance(an, bn, 12)
    if d == 1 and am == bm:
        return 1.0
    if d == 1 and am != bm:
        return 1.5
    if d == 2 and am == bm:
        return 2.0
    return 3.0


def parse_duration_to_seconds(d: str) -> Optional[int]:
    """
    Parse '6:35' -> 395 seconds. Returns None if parse fails.
    """
    if not isinstance(d, str):
        return None
    d = d.strip()
    if not d:
        return None
    try:
        parts = d.split(":")
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + int(s)
        elif len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + int(s)
        else:
            return None
    except Exception:
        return None


# ---------------------------
# Scoring & ordering
# ---------------------------
def transition_cost(a: pd.Series,
                    b: pd.Series,
                    w_key: float = 2.0,
                    w_bpm: float = 0.15,
                    w_energy: float = 0.6,
                    mode: str = "build") -> float:
    """
    Compute the cost of transitioning from track 'a' to track 'b'.
      - w_key: weight for harmonic compatibility (lower is better)
      - w_bpm: weight per BPM difference (absolute)
      - w_energy: penalty for going against the chosen arc
          * mode='build'    penalizes energy drops (we want gradual rise)
          * mode='cooldown' penalizes energy increases (we want gradual fall)
    """
    # Key compatibility
    key_cost = camelot_cost(a["Camelot"], b["Camelot"])

    # BPM smoothness
    bpm_cost = abs(float(b["BPM"]) - float(a["BPM"]))

    # Energy arc
    ea, eb = float(a.get("Energy", 0)), float(b.get("Energy", 0))
    if mode == "build":
        energy_penalty = max(0.0, ea - eb)  # penalize drops
    else:
        energy_penalty = max(0.0, eb - ea)  # penalize rises

    return w_key * key_cost + w_bpm * bpm_cost + w_energy * energy_penalty


def choose_start_track(df: pd.DataFrame,
                       prefer_low_bpm: bool = True,
                       prefer_low_energy: bool = True) -> int:
    """
    Heuristic start:
      - Prefer lowest BPM and lower energy to allow a natural build.
    Returns the row index (position in df, not the DataFrame index label).
    """
    scored = df.copy()
    scored["_bpm_rank"] = scored["BPM"].rank(method="first", ascending=True)
    scored["_energy_rank"] = scored["Energy"].rank(method="first", ascending=True)
    weight_bpm = 1.0 if prefer_low_bpm else 0.0
    weight_energy = 1.0 if prefer_low_energy else 0.0
    scored["_start_score"] = weight_bpm * scored["_bpm_rank"] + weight_energy * scored["_energy_rank"]
    # Return positional index of the minimum score
    pos = int(scored["_start_score"].astype(float).idxmin())
    # Convert label to positional index
    return df.index.get_loc(pos)


def greedy_order(df: pd.DataFrame,
                 start_pos: int,
                 mode: str,
                 w_key: float,
                 w_bpm: float,
                 w_energy: float) -> List[int]:
    """
    Greedy nearest-neighbor ordering using the transition_cost.
    Returns a list of positional indices in the chosen order.
    """
    remaining = list(range(len(df)))
    order: List[int] = []
    current = start_pos
    order.append(current)
    remaining.remove(current)

    while remaining:
        # Evaluate cost to every remaining track
        costs = []
        for j in remaining:
            c = transition_cost(df.iloc[current], df.iloc[j],
                                w_key=w_key, w_bpm=w_bpm, w_energy=w_energy, mode=mode)
            # Tiebreakers favor higher danceability, then popularity, then happiness
            tiebreak = (
                -float(df.iloc[j].get("Danceability", 0)),
                -float(df.iloc[j].get("Popularity", 0)),
                -float(df.iloc[j].get("Happiness", 0)),
            )
            costs.append((c, tiebreak, j))
        costs.sort(key=lambda x: (x[0], x[1]))  # sort by cost then tiebreakers
        current = costs[0][2]
        order.append(current)
        remaining.remove(current)

    return order


def format_track(row: pd.Series, idx: int) -> str:
    return (f"{idx:>2}. {row['Artist']} — {row['Title']}  "
            f"(Camelot {row['Camelot']}, {row['BPM']} BPM, Energy {row['Energy']})")


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Order tracks by harmonic (Camelot), BPM smoothness, and energy arc.")
    ap.add_argument("infile", help="Input file (CSV or TSV). Headers must include: Artist, Title, Camelot, BPM, Energy.")
    ap.add_argument("--sep", default=None,
                    help="Field separator. Default: auto-detect (pandas). For TSV use '\\t'.")
    ap.add_argument("--mode", choices=["build", "cooldown"], default="build",
                    help="Energy arc preference: 'build' (start low, build up) or 'cooldown' (start high, come down).")
    ap.add_argument("--start", default=None,
                    help="Optional regex to pick a start track using 'Artist|Title' match. Example: 'Martin Roth|Deep Style'")
    ap.add_argument("--w_key", type=float, default=2.0, help="Weight for key compatibility.")
    ap.add_argument("--w_bpm", type=float, default=0.15, help="Weight per BPM difference.")
    ap.add_argument("--w_energy", type=float, default=0.6, help="Weight for energy arc penalty.")
    ap.add_argument("--out", default=None, help="Optional path to write ordered CSV.")
    args = ap.parse_args()

    # Load
    df = pd.read_csv(args.infile, sep=args.sep, engine="python")

    # Basic sanitation / coercion
    required = ["Artist", "Title", "Camelot", "BPM", "Energy"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    # Coerce numeric
    for col in ["BPM", "Energy", "Danceability", "Happiness", "Popularity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Duration" in df.columns:
        # Add seconds for convenience
        df["DurationSeconds"] = df["Duration"].apply(parse_duration_to_seconds)

    # Validate Camelot
    for i, val in enumerate(df["Camelot"].astype(str).fillna("")):
        try:
            parse_camelot(val)
        except ValueError as e:
            raise SystemExit(f"Row {i} invalid Camelot code {val!r}: {e}")

    # Choose start
    if args.start:
        # regex over "Artist|Title"
        mask = df.apply(lambda r: re.search(args.start, f"{r['Artist']}|{r['Title']}", re.IGNORECASE) is not None, axis=1)
        if not mask.any():
            raise SystemExit(f"--start pattern did not match any rows: {args.start!r}")
        start_pos = df.index.get_loc(df[mask].index[0])
    else:
        start_pos = choose_start_track(df, prefer_low_bpm=(args.mode == "build"),
                                       prefer_low_energy=(args.mode == "build"))

    order_positions = greedy_order(df, start_pos, args.mode, args.w_key, args.w_bpm, args.w_energy)

    # Output
    print("\n🎧 Ordered Playlist")
    print(f"Mode: {args.mode}  |  Weights: w_key={args.w_key}, w_bpm={args.w_bpm}, w_energy={args.w_energy}\n")
    total_secs = 0
    for idx, pos in enumerate(order_positions, start=1):
        row = df.iloc[pos]
        print(format_track(row, idx))
        total_secs += int(row.get("DurationSeconds") or 0)

    if total_secs:
        h = total_secs // 3600
        m = (total_secs % 3600) // 60
        s = total_secs % 60
        print(f"\nTotal duration: {h:d}:{m:02d}:{s:02d}")

    # Optionally write CSV with an Order column
    if args.out:
        df_out = df.copy()
        # Map positional indices to 1..N
        order_map = {pos: i+1 for i, pos in enumerate(order_positions)}
        df_out["Order"] = [order_map[i] for i in range(len(df_out))]
        df_out = df_out.sort_values("Order")
        df_out.to_csv(args.out, index=False)
        print(f"\nSaved ordered CSV -> {args.out}")


if __name__ == "__main__":
    main()

