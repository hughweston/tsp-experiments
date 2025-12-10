import os
import json
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)


def calculate_tour_length(points, tour_indices):
    """
    Calculate total tour length.

    Args:
        points: List of point dictionaries with 'x' and 'y'
        tour_indices: List of point indices in visit order

    Returns:
        Total tour length
    """
    if not tour_indices:
        return 0.0

    total = 0.0
    for i in range(len(tour_indices)):
        current = points[tour_indices[i]]
        next_point = points[tour_indices[(i + 1) % len(tour_indices)]]
        total += calculate_distance(current, next_point)

    return total


def verify_solution(tsp_file, mapping_file, vlm_solution, verbose=True):
    """
    Verify a VLM solution against the optimal tour.

    Args:
        tsp_file: Path to original TSP file (points in optimal order)
        mapping_file: Path to label mapping JSON file
        vlm_solution: List of labels in VLM's proposed tour order
        verbose: Whether to print detailed output

    Returns:
        Dictionary with verification results
    """
    # Load original points (in optimal tour order)
    with open(tsp_file, 'r') as f:
        points = json.load(f)

    # Load label mapping
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)

    label_to_index = {int(k): v for k, v in mapping_data['label_to_index'].items()}

    # Validate VLM solution
    n_points = len(points)
    expected_labels = set(range(1, n_points + 1))

    # Check if solution has correct length
    if len(vlm_solution) != n_points:
        if len(vlm_solution) == n_points + 1 and vlm_solution[0] == vlm_solution[-1]:
            return {
                'valid': False,
                'error': f"Solution has {len(vlm_solution)} points. Do NOT repeat the first point at the end - the tour automatically closes.",
                'vlm_tour_length': None,
                'optimal_tour_length': None,
                'difference': None,
                'percentage_increase': None
            }
        else:
            return {
                'valid': False,
                'error': f"Solution has {len(vlm_solution)} points, expected {n_points}",
                'vlm_tour_length': None,
                'optimal_tour_length': None,
                'difference': None,
                'percentage_increase': None
            }

    # Check if all labels are present exactly once
    vlm_labels_set = set(vlm_solution)

    if vlm_labels_set != expected_labels:
        missing = expected_labels - vlm_labels_set
        extra = vlm_labels_set - expected_labels
        error_msg = []
        if missing:
            error_msg.append(f"Missing labels: {sorted(missing)}")
        if extra:
            error_msg.append(f"Extra/invalid labels: {sorted(extra)}")
        return {
            'valid': False,
            'error': '; '.join(error_msg),
            'vlm_tour_length': None,
            'optimal_tour_length': None,
            'difference': None,
            'percentage_increase': None
        }

    # Convert VLM labels to point indices
    try:
        vlm_tour_indices = [label_to_index[label] for label in vlm_solution]
    except KeyError as e:
        return {
            'valid': False,
            'error': f"Invalid label in solution: {e}",
            'vlm_tour_length': None,
            'optimal_tour_length': None,
            'difference': None,
            'percentage_increase': None
        }

    # Calculate tour lengths
    vlm_tour_length = calculate_tour_length(points, vlm_tour_indices)
    optimal_tour_length = calculate_tour_length(points, list(range(n_points)))

    # Calculate difference
    difference = vlm_tour_length - optimal_tour_length
    percentage_increase = (difference / optimal_tour_length) * 100 if optimal_tour_length > 0 else 0

    # Check if solution is optimal (within floating point tolerance)
    is_optimal = abs(difference) < 0.01

    results = {
        'valid': True,
        'is_optimal': is_optimal,
        'vlm_tour_length': vlm_tour_length,
        'optimal_tour_length': optimal_tour_length,
        'difference': difference,
        'percentage_increase': percentage_increase,
        'vlm_tour_indices': vlm_tour_indices,
        'vlm_tour_labels': vlm_solution
    }

    if verbose:
        print("\n" + "="*60)
        print("TSP Solution Verification")
        print("="*60)
        print(f"TSP Instance: {Path(tsp_file).name}")
        print(f"Number of points: {n_points}")
        print(f"\nVLM Solution (labels): {vlm_solution}")
        print(f"VLM Solution (indices): {vlm_tour_indices}")
        print(f"\n{'Metric':<30} {'Value':>20}")
        print("-"*60)
        print(f"{'VLM Tour Length':<30} {vlm_tour_length:>20.2f}")
        print(f"{'Optimal Tour Length':<30} {optimal_tour_length:>20.2f}")
        print(f"{'Difference':<30} {difference:>20.2f}")
        print(f"{'Percentage Increase':<30} {percentage_increase:>19.2f}%")
        print("-"*60)

        if is_optimal:
            print("✓ OPTIMAL SOLUTION FOUND!")
        else:
            print(f"✗ Suboptimal by {difference:.2f} units ({percentage_increase:.2f}%)")
        print("="*60 + "\n")

    return results
# get gemini token output
gemini_points = []
for entry in os.listdir("tokens"):
    for file in os.listdir(f"tokens/{entry}"):
        name = file[:-4]
        if name[-3:] == "put":
            with open(f"tokens/{entry}/{file}", "r") as f:
                solution = [int(x.strip()) for x in f.readlines()[1].strip().split(',')]

            name = name[:-7]
            tsp_file = f"optimal-tsps/{name}"
            mapping_file = f"labeled_plots/{entry}/{name}_mapping.json"

            output = verify_solution(tsp_file=tsp_file, mapping_file=mapping_file, vlm_solution=solution, verbose=False)
            gemini_points.append((int(entry), output["percentage_increase"]))
gx, gy = zip(*gemini_points)

qwen_points = []
for entry in os.listdir("qwen_tokens"):
    for file in os.listdir(f"qwen_tokens/{entry}"):
        name = file[:-4]
        if name[-3:] == "put":
            with open(f"qwen_tokens/{entry}/{file}", "r") as f:
                solution = [int(x.strip()) for x in f.readlines()[1].strip().split(',')]

            name = name[:-7]
            tsp_file = f"optimal-tsps/{name}"
            mapping_file = f"labeled_plots/{entry}/{name}_mapping.json"

            output = verify_solution(tsp_file=tsp_file, mapping_file=mapping_file, vlm_solution=solution, verbose=False)
            if output["percentage_increase"] < 100:
                qwen_points.append((int(entry), output["percentage_increase"]))
qx, qy = zip(*qwen_points)

gx, gy = np.array(gx), np.array(gy)
qx, qy = np.array(qx), np.array(qy)

gm, gb = np.polyfit(gx, gy, 1)
g_fit_line = gm * gx + gb

g_slope, g_intercept, g_r_value, g_p_value, g_std_err = stats.linregress(gx, gy)
g_r_squared = g_r_value**2

qm, qb = np.polyfit(qx, qy, 1)
q_fit_line = qm * qx + qb

q_slope, q_intercept, q_r_value, q_p_value, q_std_err = stats.linregress(qx, qy)
q_r_squared = q_r_value**2


plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(10, 7))

# 1. Scatter Plots with Labels
plt.scatter(gx, gy, c="blue", label="gemini-2.5-flash")
plt.scatter(qx, qy, c="orange", label="qwen3-vl")

# 2. Line of Best Fit Plots with Labels
# Use the same color as the scatter points
plt.plot(gx, g_fit_line, 
         color='blue', 
         linestyle='--', 
         linewidth=2, 
         label=f'Gemini Fit: y={gm:.2f}x + {gb:.0f}, ($R^2={g_r_squared:.3f}$)')

plt.plot(qx, q_fit_line, 
         color='orange', 
         linestyle='-', 
         linewidth=2, 
         label=f'Qwen Fit: y={qm:.2f}x + {qb:.0f}, ($R^2={q_r_squared:.3f}$)')


plt.xticks([10, 15, 20, 25, 30])
plt.xlabel("Number of Points (TSP Problem Size)")
plt.ylabel("Percentage from Optimal")
plt.title("Size of TSP Problem vs. Percentage from Optimal")

plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("optimality.svg")