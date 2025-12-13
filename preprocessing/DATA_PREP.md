# Data Preprocessing

## Problem Metadata

Generate `problem_metadata.csv` with classifications and optimal lengths:

```bash
uv run python preprocessing/add_missing_metadata.py
```

**Output format:**
```csv
problem_id,num_points,group,hull_points,optimal_length
05b013ef-724d-407f-9591-18a4103b8040,25,clustered,11,2582.3
```

- Calculates optimal tour length from provided optimal tour order (`optimal-tsps/`)
- Classifies as clustered/dispersed using z-score metric (nearest-neighbor distances)
- Maintains balanced 36/36 split

## Human Baseline

Calculate reproducible human performance metrics:

```bash
uv run python preprocessing/calculate_human_baseline.py
```

**Output:**
```
Random baseline (one human per problem):
  Mean zero-crossings rate: 79.89%
  95% CI: [65.62%, 93.75%]
```

Uses bootstrap sampling (seed=42, 1000 iterations) to randomly select one human solution per problem, matching VLM evaluation methodology.
