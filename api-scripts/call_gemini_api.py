#!/usr/bin/env python3
"""
Call Google Gemini API on all TSP PNG images and save responses.

Set GEMINI_API_KEY environment variable.
"""

import base64
import os
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

# TSP PROMPT
PROMPT = """You are given an image showing a Traveling Salesman Problem (TSP) instance. The image contains:
- Black dots representing cities/points
- Each point is labeled with a unique integer (1 to N)
- Labels are positioned adjacent to their corresponding points

Your task is to find the shortest tour that visits all points exactly once and returns to the starting point.

Please provide your answer as a comma-separated list of the point labels in the order they should be visited.

Format: List each point label once, in visit order. Do NOT repeat the first point at the end.
Example: 1, 5, 3, 7, 2, 4, 6

You can start from any point - the tour is a cycle so starting position doesn't affect the total distance."""

# CLUSTERING PROMPT
# PROMPT = """You are given an image with labeled points (black dots with integer labels).

# Group these points into clusters based on spatial proximity. Each point must belong to exactly one cluster. You must create at least 2 clusters (cannot group all points together).

# Format your answer as:
# Cluster 1: [point labels]
# Cluster 2: [point labels]
# ...

# Example:
# Cluster 1: 1, 5, 3
# Cluster 2: 7, 2, 4
# Cluster 3: 6, 8, 9, 10"""


def main():
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Set it with: export GEMINI_API_KEY='your-key-here'")
        return 1

    # Create Gemini client
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})

    # Setup directories
    input_dir = Path("vlm-inputs")
    output_dir = Path("vlm-outputs/solutions/gemini")
    # output_dir = Path("vlm-outputs/clusters/gemini")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all PNG files
    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        print(f"No PNG files found in {input_dir}/")
        return 1

    print(f"Found {len(png_files)} PNG files")
    print(f"Saving responses to {output_dir}/\n")

    # Process each PNG
    for i, png_path in enumerate(png_files, 1):
        problem_id = png_path.stem
        output_path = output_dir / f"{problem_id}.txt"

        # Skip if already processed
        if output_path.exists():
            print(f"[{i}/{len(png_files)}] Skipping {problem_id} (already exists)")
            continue

        print(f"[{i}/{len(png_files)}] Processing {problem_id}...", end=" ", flush=True)

        try:
            # Read and encode image as base64
            with open(png_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Generate response using inline_data
            response = client.models.generate_content(
                model="gemini-3-pro-preview",
                contents=[
                    types.Content(
                        parts=[
                            types.Part(text=PROMPT),
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/png",
                                    data=base64.b64decode(image_data),
                                )
                            ),
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="low")
                ),
            )

            # Save response
            with open(output_path, "w") as f:
                f.write(response.text)

            print("✓ Saved")
            time.sleep(5)

        except Exception as e:
            print(f"✗ Error: {e}")
            # Save error to file
            with open(output_path, "w") as f:
                f.write(f"ERROR: {e}")

    print(f"\n✓ Complete! Responses saved to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
