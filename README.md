# TSP-VLM Experiments

Testing Vision-Language Models on Traveling Salesman Problems with randomized point labels.

## Overview

Generate labeled TSP visualizations in PNG or SVG format and verify VLM solutions against optimal tours. Point labels are **randomized** so VLMs cannot rely on sequential patterns and must actually solve the spatial optimization problem.

## Project Files

**Generation & Verification:**
- `create_labeled_tsps.py` - Generate labeled images for all TSP files in a directory
- `verify_tsp_solution.py` - Verify VLM solutions against optimal tours

## Quick Start

```bash
# Generate all labeled SVG images
uv run python batch_plot_all.py --seed 42

# Generate a single file
uv run python plot_tsp_svg_labels.py optimal-tsps/[filename] -o output.svg --seed 42

# Verify a VLM solution
uv run python verify_tsp_solution.py \
  optimal-tsps/[filename] \
  vlm-inputs/[filename]_mapping.json \
  "1,5,3,7,2,4,6"
```

## Usage

### Generate Labeled Images

```bash
# Generate PNGs (default: optimal-tsps/ → vlm-inputs/)
uv run python create_labeled_tsps.py --seed 42

# Custom directories or SVG format
uv run python create_labeled_tsps.py -i optimal-tsps -o vlm-inputs -f svg
```

**Output:** For each TSP file, generates:
- `[filename].png` (or `.svg`) - Labeled visualization for VLM
- `[filename]_mapping.json` - Label-to-index mapping for verification

### Verify VLM Solutions

```bash
uv run python verify_tsp_solution.py \
  optimal-tsps/[filename] \
  vlm-inputs/[filename]_mapping.json \
  "9,8,1,10,3,7,6,5,2,4"
```

**Important:**
- Spaces in list are OK: `"1,2,3"` and `"1, 2, 3"` both work
- Starting point doesn't matter (tours are cycles)
- **Do NOT** repeat first point: use `"1,2,3"` NOT `"1,2,3,1"`

## VLM Prompt Template

```
You are given an image showing a Traveling Salesman Problem (TSP) instance. The image contains:
- Black dots representing cities/points
- Each point is labeled with a unique integer (1 to N)
- Labels are positioned adjacent to their corresponding points

Your task is to find the shortest tour that visits all points exactly once and returns to the starting point.

Please provide your answer as a comma-separated list of the point labels in the order they should be visited.

Format: List each point label once, in visit order. Do NOT repeat the first point at the end.
Example: 1, 5, 3, 7, 2, 4, 6

You can start from any point - the tour is a cycle so starting position doesn't affect the total distance.
```

## Options

**create_labeled_tsps.py:**
- `-i, --input-dir` - Input directory (default: optimal-tsps)
- `-o, --output-dir` - Output directory (default: vlm-inputs)
- `-f, --format` - Output format: png or svg (default: png)
- `-s, --seed` - Base random seed (each file gets seed+index)
- `--show-tour` - Show tour lines in all plots
- `--label-offset` - Distance of label from point (default: 20)
- `--label-font-size` - Font size for labels (default: 14)

**verify_tsp_solution.py:**
- `-q, --quiet` - Suppress detailed output (only return exit code)

## Design Notes

**Image Output:** 800×500px, black circles (radius 5), white background, plain black text labels (no backgrounds). Supports PNG and SVG formats.

**Label Positioning:** Fixed 20px distance with smart direction selection (tests 8 positions, avoids occlusion, ensures bounds).

**Randomization:** Labels (1 to N) shuffled with reproducible seed to prevent VLMs from exploiting sequential patterns.

**Verification:** Euclidean distance calculation, rotation-invariant comparison, exit code 0 for optimal / 1 for suboptimal.
