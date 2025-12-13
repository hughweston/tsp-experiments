# Metrics and Analysis

Based on **Marupudi et al. (2022)**: "Use of clustering in human solutions of the traveling salesperson problem"

## Core Metrics

**1. Cluster Deviance**
```
cluster_deviance = (t - c) / (n - c)
```
- `t` = cross-cluster transitions, `c` = number of clusters, `n` = total points
- **0.0** = perfect congruence, **1.0** = maximum deviance
- **Human baseline:** 52% achieve perfect congruence (deviance = 0)

**2. Convex Hull Adherence**
- `hull_contiguous`: Boolean - all hull points in one segment
- `hull_switches`: Number of hull/interior transitions
- `adherence_score`: 0-1 score (1.0 = perfect)

**3. Path Crossings**
- Number of edge intersections (optimal tours have zero)
- **Human baseline:** 79.9% zero-crossings rate (32 problems, seed=42)

**4. Temporal Dynamics** (humans only, requires click data)
- 50+ ms longer for between-cluster connections

## Individual Heuristic Scripts

**Clustering:**
```bash
uv run python analysis/heuristics/clustering.py \
  vlm-outputs/clusters/model_clusters.json \
  vlm-outputs/solutions/model_solutions.json
```

**Convex Hull (VLM):**
```bash
uv run python analysis/heuristics/convex_hull_vlm.py \
  vlm-outputs/solutions/model_solutions.json
```

**Convex Hull (Human):**
```bash
uv run python analysis/heuristics/convex_hull_human.py \
  marupudi_data/tsp-trials-anonymized.json \
  --group clustered --output human_hull.csv
```

**Crossings:**
```bash
uv run python analysis/heuristics/crossings.py \
  vlm-outputs/solutions/model_solutions.json
```

## Visualizations

The `create_plots.py` script generates:
- Perfect cluster congruence prevalence
- Performance by problem size (10-30 points)
- Hull adherence distributions
- Cluster deviance comparisons (VLM vs baselines)
- Crossing analysis and summary statistics
