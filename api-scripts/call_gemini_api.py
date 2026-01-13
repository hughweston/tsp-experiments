#!/usr/bin/env python3
"""
Call Google Gemini API on all TSP PNG images and save responses.

Set GEMINI_API_KEY environment variable.

Usage:
  python call_gemini_api.py --mode solution  # For TSP solutions
  python call_gemini_api.py --mode cluster   # For clustering
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

# TSP SOLUTION PROMPT
TSP_PROMPT = """You are given an image showing a Traveling Salesman Problem (TSP) instance with labeled points (1 to N).
The problem will have exactly 10, 15, 20, 25, or 30 points.

Your task is to find the shortest tour that visits all points exactly once and returns to the starting point.
You can start from any point - the tour is a cycle so starting position doesn't affect the total distance.

IMPORTANT: On your FINAL LINE, output ONLY the comma-separated list of point labels in visit order.
- Include all N points exactly once (no repeats, no omissions)
- Do NOT repeat the first point at the end
- Use plain text only (no bold, no markdown)

Example for 10 points: 1, 5, 3, 7, 2, 4, 6, 9, 8, 10"""

# CLUSTERING PROMPT
CLUSTER_PROMPT = """You are given an image with labeled points (black dots with integer labels).
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
Cluster 3: 6, 9, 8, 10"""


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Call Gemini API for TSP solutions or clustering"
    )
    parser.add_argument(
        "--mode",
        choices=["solution", "cluster"],
        required=True,
        help="Mode: 'solution' for TSP solutions, 'cluster' for clustering",
    )
    args = parser.parse_args()

    # Select prompt and output directory based on mode
    if args.mode == "solution":
        prompt = TSP_PROMPT
        output_dir = Path("vlm-outputs/solutions/gemini")
    else:  # cluster
        prompt = CLUSTER_PROMPT
        output_dir = Path("vlm-outputs/clusters/gemini")

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
            if args.mode == "solution":
                response = client.models.generate_content(
                    model="gemini-3-pro-preview",
                    contents=[
                        types.Content(
                            parts=[
                                types.Part(text=prompt),
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
                        thinking_config=types.ThinkingConfig(thinking_level="high"),
                    ),
                )
            else:  # cluster mode
                response = client.models.generate_content(
                    model="gemini-3-pro-preview",
                    contents=[
                        types.Content(
                            parts=[
                                types.Part(text=prompt),
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
                        thinking_config=types.ThinkingConfig(thinking_level="low"),
                        max_output_tokens=2048,
                    ),
                )

            # Save response
            with open(output_path, "w") as f:
                f.write(response.text or "")

            # Save token metadata to JSONL (only for solution mode)
            if args.mode == "solution":
                token_file = output_dir.parent / f"{output_dir.name}_tokens.jsonl"
                usage = response.usage_metadata
                
                # Extract detailed token counts
                thinking_tokens = getattr(usage, "thoughts_token_count", 0)
                if thinking_tokens == 0:
                    thinking_tokens = getattr(usage, "thinking_tokens", 0)

                token_data = {
                    "problem_id": problem_id,
                    "usage": {
                        "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                        "candidates_tokens": getattr(usage, "candidates_token_count", 0),
                        "total_tokens": getattr(usage, "total_token_count", 0),
                        "thinking_tokens": thinking_tokens
                    }
                }

                with open(token_file, "a") as f:
                    f.write(json.dumps(token_data) + "\n")
                    f.flush()

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
