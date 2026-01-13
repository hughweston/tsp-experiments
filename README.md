# TSP-VLM Experiments

Testing Vision-Language Models on Traveling Salesman Problems with randomized point labels.

## Overview

Generate labeled TSP visualizations in PNG or SVG format and verify VLM solutions against optimal tours. Point labels are **randomized** so VLMs cannot rely on sequential patterns and must actually solve the spatial optimization problem.

## Project Structure

**Analysis:** `analysis/` - See [METRICS.md](analysis/METRICS.md) for details
- `analyze_all.py` - Complete analysis pipeline
- `compare_models.py` - Multi-model comparison
- `create_plots.py` - Generate visualizations
- `heuristics/` - Clustering, hull adherence, crossings metrics

**Preprocessing:** `preprocessing/` - See [DATA_PREP.md](preprocessing/DATA_PREP.md) for details
- `add_missing_metadata.py` - Generate problem_metadata.csv
- `calculate_human_baseline.py` - Human performance baselines

**Data:**
- `marupudi_data/` - Human performance data (tsp-singles.csv, tsp-trials-anonymized.json)
- `problem_metadata.csv` - Problem classifications and optimal lengths
- `optimal-tsps/` - TSP problems in optimal tour order
- `vlm-inputs/` - Generated labeled PNGs and mappings
- `vlm-outputs/` - VLM solutions and clusters (raw text + parsed JSON)
- `results/` - Analysis outputs (auto-generated)

**Tools:**
- `create_labeled_tsps.py` - Generate labeled images
- `verify_tsp_solution.py` - Verify VLM solutions
- `api-scripts/` - Automated API calls and parsing

## Quick Start - Analyze VLM Results

**Note:** All commands should be run from the repository root directory.

**Analyze a single model:**

```bash
# Analyzes and creates results/claude_opus_4.5_solutions/
uv run python analysis/analyze_all.py vlm-outputs/solutions/claude_opus_4.5_solutions.json \
  --vlm-clusters vlm-outputs/clusters/claude_opus_4.5_clusters.json
```

This automatically:
- ✓ Measures convex hull adherence, path crossings, and cluster deviance
- ✓ Combines all metrics with problem metadata (clustered/disperse groups)
- ✓ Saves results to `results/[model_name]/[model_name].csv`
- ✓ Generates all visualizations in `results/[model_name]/visualizations/`

**Compare multiple models:**

```bash
# Analyze each model (repeat for each)
uv run python analysis/analyze_all.py vlm-outputs/solutions/[model]_solutions.json \
  --vlm-clusters vlm-outputs/clusters/[model]_clusters.json

# Generate cross-model comparisons
uv run python analysis/compare_models.py
```

## Complete Workflow

### 1. Generate labeled TSP images for VLMs
```bash
uv run python create_labeled_tsps.py --seed 42
```

### 2. Get VLM solutions

**Option A: Automated API calls**
```bash
# Set API keys
export GEMINI_API_KEY='your-gemini-key'
export ANTHROPIC_API_KEY='your-claude-key'
export OPENAI_API_KEY='your-openai-key'

# Call APIs for TSP solutions (--mode solution)
uv run python api-scripts/call_gemini_api.py --mode solution    # → vlm-outputs/solutions/gemini/*.txt
uv run python api-scripts/call_claude_api.py --mode solution    # → vlm-outputs/solutions/claude/*.txt
uv run python api-scripts/call_openai_api.py --mode solution    # → vlm-outputs/solutions/openai/*.txt

# Parse API text responses into JSON solution files
uv run python api-scripts/parse_api_solutions.py vlm-outputs/solutions/claude vlm-outputs/solutions/claude_opus_4.5_solutions.json
uv run python api-scripts/parse_api_solutions.py vlm-outputs/solutions/openai vlm-outputs/solutions/gpt_5.2_solutions.json
uv run python api-scripts/parse_api_solutions.py vlm-outputs/solutions/gemini vlm-outputs/solutions/gemini_3_pro_solutions.json

# Call APIs for clustering (--mode cluster)
uv run python api-scripts/call_gemini_api.py --mode cluster    # → vlm-outputs/clusters/gemini/*.txt
uv run python api-scripts/call_claude_api.py --mode cluster    # → vlm-outputs/clusters/claude/*.txt
uv run python api-scripts/call_openai_api.py --mode cluster    # → vlm-outputs/clusters/openai/*.txt

# Parse cluster outputs into JSON cluster files
uv run python api-scripts/parse_api_clusters.py vlm-outputs/clusters/claude vlm-outputs/clusters/claude_opus_4.5_clusters.json
uv run python api-scripts/parse_api_clusters.py vlm-outputs/clusters/openai vlm-outputs/clusters/gpt_5.2_clusters.json
uv run python api-scripts/parse_api_clusters.py vlm-outputs/clusters/gemini vlm-outputs/clusters/gemini_3_pro_clusters.json
```

**Option B: Manual collection**
Present images to your VLM and collect solutions in JSON format.

Solution JSON format (keys match TSP filenames without extension):
```json
{
  "0fed7bdc-a343-4966-b202-69661c45fdb2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  "2549fb3d-39ec-4ed0-86af-bb0f2f473e68": [5, 3, 1, 4, 2, 6, 8, 7, 9, 10]
}
```

### 3. Analyze VLM performance

```bash
uv run python analysis/analyze_all.py vlm-outputs/solutions/claude_opus_4.5_solutions.json \
  --vlm-clusters vlm-outputs/clusters/claude_opus_4.5_clusters.json
```

See [analysis/METRICS.md](analysis/METRICS.md) for individual heuristic scripts and metric details.

## Quick Start (Single Solutions)

```bash
# Verify a single VLM solution
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

### TSP Solution Prompt

```
You are given an image showing a Traveling Salesman Problem (TSP) instance with labeled points (1 to N).
The problem will have exactly 10, 15, 20, 25, or 30 points.

Your task is to find the shortest tour that visits all points exactly once and returns to the starting point.
You can start from any point - the tour is a cycle so starting position doesn't affect the total distance.

IMPORTANT: On your FINAL LINE, output ONLY the comma-separated list of point labels in visit order.
- Include all N points exactly once (no repeats, no omissions)
- Do NOT repeat the first point at the end
- Use plain text only (no bold, no markdown)

Example for 10 points: 1, 5, 3, 7, 2, 4, 6, 9, 8, 10
```

### Clustering Prompt

```
You are given an image with labeled points (black dots with integer labels).
The image contains exactly 10, 15, 20, 25, or 30 points.

Group these points into clusters based on spatial proximity. Each point must belong to exactly one cluster.
You must create at least 2 clusters (cannot group all points together).

IMPORTANT: Format your answer exactly as shown below with plain text only (no bold, no markdown):
Cluster 1: [comma-separated point labels]
Cluster 2: [comma-separated point labels]
...

Include all N points exactly once (no repeats, no omissions).

Example for 10 points:
Cluster 1: 1, 5, 3
Cluster 2: 7, 2, 4
Cluster 3: 6, 9, 8, 10
```

## API Configuration

All models use consistent settings for fair comparison:

| Model | Thinking Level | Output Limit |
|-------|---------------|--------------|
| Claude Opus 4.5 | Extended thinking (50k budget) | 64000 (max) |
| GPT-5.2 | Reasoning effort: high | 128000 (max) |
| Gemini 3 Pro | Thinking level: high | 65356 (max) |

The prompt includes explicit instruction for final-line format to ensure reliable parsing.

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

## Documentation

- [analysis/METRICS.md](analysis/METRICS.md) - Metric definitions, heuristic scripts, visualizations
- [preprocessing/DATA_PREP.md](preprocessing/DATA_PREP.md) - Metadata generation, human baselines
