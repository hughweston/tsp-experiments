#!/usr/bin/env python3
"""
Parse API cluster outputs and convert to JSON format.

Reads raw cluster text outputs from VLMs and extracts cluster assignments
into a structured JSON format.
"""

import csv
import json
import sys
from pathlib import Path


def parse_cluster_file(cluster_file):
    """
    Parse VLM cluster output from file into dict of {point_label: cluster_id}.

    Handles multiple formats:
    - Cluster 1: 1, 5, 3
    - **Cluster 1:** 1, 5, 3
    - **Cluster 1: Description**
      1, 5, 3
    """
    import re

    with open(cluster_file) as f:
        content = f.read()

    # Check for error files
    if content.startswith("ERROR:"):
        return None

    assignments = {}
    cluster_id = 0

    # Split by lines
    lines = content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Strip markdown formatting
        line_clean = line.replace("**", "").replace("*", "").strip()

        # Look for lines that contain "Cluster" and ":"
        if "cluster" in line_clean.lower() and ":" in line_clean:
            # Extract everything after "Cluster X:"
            parts = line_clean.split(":", 1)
            if len(parts) >= 2:
                points_str = parts[1].strip()

                # If points_str is empty or has no digits, look ahead up to 5 lines
                if not points_str or not any(c.isdigit() for c in points_str):
                    # Look ahead for points (up to 5 lines or until next cluster)
                    for j in range(1, min(6, len(lines) - i)):
                        next_line = lines[i + j].strip()
                        next_line_clean = (
                            next_line.replace("**", "").replace("*", "").strip()
                        )

                        # Stop if we hit another cluster header
                        if (
                            "cluster" in next_line_clean.lower()
                            and ":" in next_line_clean
                        ):
                            break

                        # Check if this line has comma-separated numbers (and is short enough to be a cluster list)
                        if (
                            "," in next_line
                            and any(c.isdigit() for c in next_line)
                            and len(next_line) < 100
                        ):
                            points_str = next_line
                            i += j  # Skip the lines we've consumed
                            break

                # Only parse if it looks like a cluster list (short, has commas, mostly numbers)
                # Skip verbose explanation text
                if len(points_str) > 200:  # Too long, likely explanation text
                    i += 1
                    continue

                # Check if line has too many non-numeric words (likely explanation)
                words = points_str.split()
                if len(words) > 20:  # Too many words, likely explanation
                    i += 1
                    continue

                # Parse point labels using regex to extract individual numbers
                # This avoids concatenating digits from different words
                points = []
                # Match individual numbers (optionally with brackets or parentheses)
                number_pattern = r"\b(\d+)\b"
                matches = re.findall(number_pattern, points_str)
                for match in matches:
                    num = int(match)
                    # Basic sanity check: TSP points are usually 1-40
                    if 1 <= num <= 40:
                        points.append(num)

                # Assign all points to this cluster (only if we found points)
                if points:
                    for point in points:
                        assignments[point] = cluster_id
                    cluster_id += 1

        i += 1

    # Normalize cluster IDs to start from 0 and be consecutive
    if assignments:
        unique_clusters = sorted(set(assignments.values()))
        cluster_mapping = {
            old_id: new_id for new_id, old_id in enumerate(unique_clusters)
        }
        assignments = {
            point: cluster_mapping[cluster_id]
            for point, cluster_id in assignments.items()
        }

    return assignments if assignments else None


def load_problem_metadata():
    """Load problem metadata from CSV to get expected point counts."""
    metadata = {}
    metadata_path = Path("problem_metadata.csv")

    if not metadata_path.exists():
        print(f"Warning: {metadata_path} not found, skipping validation")
        return metadata

    with open(metadata_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row["problem_id"]] = int(row["num_points"])

    return metadata


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python parse_api_clusters.py <cluster_output_dir> <output_json_file>"
        )
        print(
            "Example: python parse_api_clusters.py vlm-outputs/clusters/claude claude_clusters.json"
        )
        return 1

    cluster_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    if not cluster_dir.exists():
        print(f"Error: Directory {cluster_dir} does not exist")
        return 1

    # Load problem metadata for validation
    metadata = load_problem_metadata()

    # Process all cluster text files
    txt_files = sorted(cluster_dir.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {cluster_dir}/")
        return 1

    print(f"Processing {len(txt_files)} files from {cluster_dir}...")

    clusters = {}
    parsed_count = 0
    error_count = 0

    for txt_file in txt_files:
        problem_id = txt_file.stem

        # Parse cluster assignments
        assignments = parse_cluster_file(txt_file)

        if assignments is None:
            print(f"✗ {problem_id}: Could not parse clusters")
            error_count += 1
            # Store empty dict for failed parses
            clusters[problem_id] = {}
        else:
            # Validate number of points is a valid size (10, 15, 20, 25, or 30)
            valid_sizes = {10, 15, 20, 25, 30}
            if len(assignments) not in valid_sizes:
                print(
                    f"✗ {problem_id}: Invalid size {len(assignments)} (must be 10, 15, 20, 25, or 30) - leaving empty"
                )
                clusters[problem_id] = {}
                error_count += 1
                continue

            # Validate number of points against metadata
            expected_points = metadata.get(problem_id)
            if expected_points and len(assignments) != expected_points:
                print(
                    f"✗ {problem_id}: Expected {expected_points} points but got {len(assignments)} - leaving empty"
                )
                clusters[problem_id] = {}
                error_count += 1
            else:
                clusters[problem_id] = assignments
                num_clusters = len(set(assignments.values()))
                print(
                    f"✓ {problem_id}: {len(assignments)} points in {num_clusters} clusters"
                )
                parsed_count += 1

    # Validate distribution before saving
    from collections import Counter

    size_counts = Counter()
    for cluster_assignments in clusters.values():
        if cluster_assignments:  # Only count non-empty clusters
            size_counts[len(cluster_assignments)] += 1

    expected_distribution = {10: 16, 15: 16, 20: 16, 25: 16, 30: 8}
    distribution_error = False

    for size, expected_count in expected_distribution.items():
        actual_count = size_counts.get(size, 0)
        if actual_count != expected_count:
            distribution_error = True

    # Save to JSON with compact format
    json_str = "{\n"
    items = list(clusters.items())
    for i, (key, value) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        json_str += f'  "{key}": {json.dumps(value)}{comma}\n'
    json_str += "}"

    with open(output_file, "w") as f:
        f.write(json_str)

    print(f"\n✓ Parsed {parsed_count}/{len(txt_files)} cluster files")
    if error_count > 0:
        print(f"⚠ Failed to parse {error_count} files")
    print(f"✓ Saved to {output_file}")

    # Print distribution error if needed
    if distribution_error:
        print("\n" + "=" * 70)
        print("❌ ERROR: DISTRIBUTION MISMATCH ❌")
        print("=" * 70)
        print("\nExpected distribution:")
        print("  10 points: 16 instances")
        print("  15 points: 16 instances")
        print("  20 points: 16 instances")
        print("  25 points: 16 instances")
        print("  30 points: 8 instances")
        print("\nActual distribution:")
        for size in [10, 15, 20, 25, 30]:
            actual = size_counts.get(size, 0)
            expected = expected_distribution[size]
            status = "✓" if actual == expected else "✗"
            print(f"  {status} {size} points: {actual} instances (expected {expected})")
        print("\n" + "=" * 70)
        print("Please manually review and fix the errors above!")
        print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
