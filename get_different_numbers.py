#!/usr/bin/env python3
"""
Batch process all TSP files in the optimal-tsps folder.
Creates labeled SVG plots for each file.
"""

import os
import json
from pathlib import Path
from plot_tsp_png_labels import plot_tsp_with_random_labels


def batch_plot_all_tsps(input_dir='optimal-tsps', output_dir='labeled_plots',
                        seed=None, show_tour=False, label_offset=20,
                        label_font_size=14):
    """
    Generate labeled SVG plots for all TSP files in a directory.

    Args:
        input_dir: Directory containing TSP JSON files
        output_dir: Directory to save output plots
        seed: Random seed (if None, each plot gets a different random assignment)
        show_tour: Whether to show the tour line
        label_offset: Distance of label from point (fixed distance, smart direction)
        label_font_size: Font size for labels
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    for i in [10, 15, 20, 25, 30]:
        path = Path(output_dir + f"/{str(i)}")
        path.mkdir(exist_ok=True)


    # Get all files in input directory
    input_path = Path(input_dir)
    tsp_files = [f for f in input_path.iterdir() if f.is_file()]

    print(f"Found {len(tsp_files)} TSP files to process")
    print(f"Output directory: {output_path.absolute()}\n")

    for i, tsp_file in enumerate(tsp_files, 1):
        with open(tsp_file, 'r') as f:
            points = json.load(f)

        n_points = len(points)
        output_file = output_path / str(n_points) / f"{tsp_file.name}"

        print(f"[{i}/{len(tsp_files)}] Processing {tsp_file.name}...")

        try:
            # Use seed + file index for reproducibility while varying between files
            file_seed = seed + i if seed is not None else None

            plot_tsp_with_random_labels(
                str(tsp_file),
                output_path=str(output_file),
                seed=file_seed,
                label_offset=label_offset,
                label_font_size=label_font_size,
                show_tour=show_tour
            )
            print(f"  ✓ Saved to {output_file.name}\n")

        except Exception as e:
            print(f"  ✗ Error processing {tsp_file.name}: {e}\n")

    print(f"\nCompleted! Processed {len(tsp_files)} files.")
    print(f"All plots saved to: {output_path.absolute()}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Batch plot all TSP files with random labels (SVG format)'
    )
    parser.add_argument('-i', '--input-dir', default='optimal-tsps',
                       help='Input directory with TSP files (default: optimal-tsps)')
    parser.add_argument('-o', '--output-dir', default='labeled_plots',
                       help='Output directory for plots (default: labeled_plots)')
    parser.add_argument('-s', '--seed', type=int,
                       help='Base random seed (each file gets seed+index)')
    parser.add_argument('--show-tour', action='store_true',
                       help='Show the tour line connecting points')
    parser.add_argument('--label-offset', type=int, default=20,
                       help='Distance of label from point (default: 20)')
    parser.add_argument('--label-font-size', type=int, default=14,
                       help='Font size for labels (default: 14)')

    args = parser.parse_args()

    batch_plot_all_tsps(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        show_tour=args.show_tour,
        label_offset=args.label_offset,
        label_font_size=args.label_font_size
    )


if __name__ == '__main__':
    main()
