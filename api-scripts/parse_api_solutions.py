#!/usr/bin/env python3
"""
Parse API output files and populate solutions JSON files.
Works for Claude, OpenAI, and Gemini's outputs.
"""

import json
import re
import sys
from pathlib import Path


def extract_tour(text):
    """Extract the TSP tour from API response text."""
    # Try to find the last line with comma-separated numbers
    lines = text.strip().split("\n")

    # Work backwards from the last line
    for line in reversed(lines):
        # Remove markdown bold markers and other formatting
        line = line.replace("**", "").replace("Answer:", "").strip()

        # Skip empty lines
        if not line:
            continue

        # Look for comma-separated numbers
        # Match patterns like "1, 2, 3, 4" or "1,2,3,4"
        numbers = re.findall(r"\d+", line)

        if len(numbers) > 3:  # Need at least a few points to be a valid tour
            return [int(n) for n in numbers]

    return []


def main():
    if len(sys.argv) != 3:
        print("Usage: python parse_api_outputs.py <output_dir> <json_file>")
        print(
            "Example: python parse_api_outputs.py claude_outputs claude_opus_4.5_solutions.json"
        )
        return 1

    output_dir = Path(sys.argv[1])
    json_path = Path(sys.argv[2])

    if not output_dir.exists():
        print(f"Error: Directory {output_dir} does not exist")
        return 1

    # Load existing JSON or create empty dict
    if json_path.exists():
        with open(json_path, "r") as f:
            solutions = json.load(f)
    else:
        solutions = {}

    # Process all output files
    txt_files = sorted(output_dir.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {output_dir}/")
        return 1

    print(f"Processing {len(txt_files)} files from {output_dir}...")

    updated_count = 0
    error_count = 0

    for txt_file in txt_files:
        problem_id = txt_file.stem

        # Initialize if not in solutions
        if problem_id not in solutions:
            solutions[problem_id] = []

        # Read the output
        with open(txt_file, "r") as f:
            text = f.read()

        # Skip error files
        if text.startswith("ERROR:"):
            print(f"✗ {problem_id}: Error file, skipping")
            error_count += 1
            continue

        # Extract tour
        tour = extract_tour(text)

        if tour:
            solutions[problem_id] = tour
            updated_count += 1
            print(f"✓ {problem_id}: {len(tour)} points")
        else:
            print(f"⚠ {problem_id}: Could not extract tour")

    # Save updated JSON with lists on single lines
    json_str = "{\n"
    items = list(solutions.items())
    for i, (key, value) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        json_str += f'  "{key}": {json.dumps(value)}{comma}\n'
    json_str += "}"

    with open(json_path, "w") as f:
        f.write(json_str)

    print(f"\n✓ Updated {updated_count}/{len(txt_files)} entries")
    if error_count > 0:
        print(f"⚠ Skipped {error_count} error files")
    print(f"✓ Saved to {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
