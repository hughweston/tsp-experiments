#!/usr/bin/env python3
"""
Process all TSP files in a directory and create labeled plots.
"""

import json
import random
import math
from pathlib import Path


def find_label_position(point, all_points, offset_distance=20,
                       width=800, height=500, label_padding=15):
    """
    Find a good position for a label that avoids overlapping with nearby points
    and stays within image bounds.
    """
    x, y = point

    # Try 8 different positions around the point
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    candidates = []

    for angle in angles:
        rad = math.radians(angle)
        label_x = x + offset_distance * math.cos(rad)
        label_y = y + offset_distance * math.sin(rad)

        # Check if label is within bounds (with padding)
        in_bounds = (label_padding <= label_x <= width - label_padding and
                    label_padding <= label_y <= height - label_padding)

        # Calculate minimum distance to any other point
        min_dist = float('inf')
        for other_x, other_y in all_points:
            if (other_x, other_y) == (x, y):
                continue
            dist = math.sqrt((label_x - other_x)**2 + (label_y - other_y)**2)
            min_dist = min(min_dist, dist)

        candidates.append({
            'x': label_x,
            'y': label_y,
            'min_dist': min_dist,
            'in_bounds': in_bounds
        })

    # First, try to find the best position that's in bounds
    in_bounds_candidates = [c for c in candidates if c['in_bounds']]

    if in_bounds_candidates:
        # Choose in-bounds position with maximum distance to other points
        best = max(in_bounds_candidates, key=lambda c: c['min_dist'])
    else:
        # No in-bounds positions, clamp the best position to bounds
        best = max(candidates, key=lambda c: c['min_dist'])
        best['x'] = max(label_padding, min(width - label_padding, best['x']))
        best['y'] = max(label_padding, min(height - label_padding, best['y']))

    return best['x'], best['y']


def create_svg_with_labels(points, labels, width=800, height=500,
                          point_radius=5, label_offset=20,
                          label_font_size=14,
                          show_tour=False, tour_order=None):
    """Create SVG content with labeled points."""
    svg_parts = []

    # SVG header
    svg_parts.append(f'<svg width="{width}" xmlns="http://www.w3.org/2000/svg" height="{height}">')
    svg_parts.append(f'  <rect width="{width}" fill="white" height="{height}" />')

    # Draw tour line if requested (before points so points appear on top)
    if show_tour and tour_order:
        path_points = [f"{points[i]['x']} {points[i]['y']}" for i in tour_order]
        path_d = "M " + " L ".join(path_points)
        svg_parts.append(f'  <path fill="none" stroke="blue" d="{path_d}" />')

    # Draw points
    for p in points:
        svg_parts.append(f'  <circle cy="{p["y"]}" cx="{p["x"]}" r="{point_radius}" />')

    # Draw labels
    all_point_coords = [(p['x'], p['y']) for p in points]

    for p, label in zip(points, labels):
        label_x, label_y = find_label_position((p['x'], p['y']), all_point_coords,
                                               label_offset, width, height)
        svg_parts.append(
            f'  <text x="{label_x:.1f}" y="{label_y:.1f}" '
            f'text-anchor="middle" dominant-baseline="central" '
            f'font-size="{label_font_size}" font-weight="bold" '
            f'font-family="Arial, sans-serif">{label}</text>'
        )

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def plot_tsp_file(file_path, output_path, width=800, height=500,
                  seed=None, point_radius=5, label_offset=20,
                  label_font_size=14, show_tour=False, format='png'):
    """Plot a single TSP file with random labels."""
    if seed is not None:
        random.seed(seed)

    # Read points from file
    with open(file_path, 'r') as f:
        points = json.load(f)

    n_points = len(points)

    # Generate random labels (1 to n_points)
    labels = list(range(1, n_points + 1))
    random.shuffle(labels)

    # Tour order is just 0, 1, 2, ... n-1 (order in the file)
    tour_order = list(range(n_points)) if show_tour else None

    # Create SVG
    svg_content = create_svg_with_labels(
        points, labels, width, height,
        point_radius, label_offset,
        label_font_size,
        show_tour, tour_order
    )

    # Save output
    if format == 'png':
        try:
            import cairosvg
            cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=output_path)
        except ImportError:
            print("Error: cairosvg not installed. Install with: uv add cairosvg")
            raise
    else:  # svg
        with open(output_path, 'w') as f:
            f.write(svg_content)

    # Save label mapping as JSON
    mapping_path = output_path.rsplit('.', 1)[0] + '_mapping.json'
    mapping = {
        'points': points,
        'labels': labels,
        'index_to_label': {i: label for i, label in enumerate(labels)},
        'label_to_index': {label: i for i, label in enumerate(labels)}
    }
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)


def create_labeled_tsps(input_dir='optimal-tsps', output_dir='vlm-inputs',
                        seed=None, show_tour=False, label_offset=20,
                        label_font_size=14, format='png',
                        width=800, height=500, point_radius=5):
    """
    Generate labeled plots for all TSP files in a directory.

    Args:
        input_dir: Directory containing TSP JSON files
        output_dir: Directory to save output plots
        seed: Random seed base (if None, each plot gets different random assignment)
        show_tour: Whether to show the tour line
        label_offset: Distance of label from point
        label_font_size: Font size for labels
        format: Output format ('png' or 'svg')
        width: Image width
        height: Image height
        point_radius: Radius of point circles
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Get all files in input directory
    input_path = Path(input_dir)
    tsp_files = sorted([f for f in input_path.iterdir() if f.is_file()])

    if not tsp_files:
        print(f"No files found in {input_dir}/")
        return

    print(f"Found {len(tsp_files)} TSP files to process")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Format: {format.upper()}\n")

    # Process each file
    for i, tsp_file in enumerate(tsp_files, 1):
        output_file = output_path / f"{tsp_file.name}.{format}"

        print(f"[{i}/{len(tsp_files)}] Processing {tsp_file.name}...", end=" ", flush=True)

        try:
            # Use seed + file index for reproducibility while varying between files
            file_seed = seed + i if seed is not None else None

            plot_tsp_file(
                str(tsp_file),
                str(output_file),
                width=width,
                height=height,
                seed=file_seed,
                point_radius=point_radius,
                label_offset=label_offset,
                label_font_size=label_font_size,
                show_tour=show_tour,
                format=format
            )
            print(f"✓ Saved {output_file.name}")

        except Exception as e:
            print(f"✗ Error: {e}")

    print(f"\n✓ Complete! Processed {len(tsp_files)} files.")
    print(f"All plots saved to: {output_path.absolute()}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Create labeled plots for all TSP files in a directory'
    )
    parser.add_argument('-i', '--input-dir', default='optimal-tsps',
                       help='Input directory with TSP files (default: optimal-tsps)')
    parser.add_argument('-o', '--output-dir', default='vlm-inputs',
                       help='Output directory for plots (default: vlm-inputs)')
    parser.add_argument('-f', '--format', choices=['png', 'svg'], default='png',
                       help='Output format: png or svg (default: png)')
    parser.add_argument('-s', '--seed', type=int,
                       help='Base random seed (each file gets seed+index)')
    parser.add_argument('--show-tour', action='store_true',
                       help='Show the tour line connecting points')
    parser.add_argument('--width', type=int, default=800,
                       help='Image width (default: 800)')
    parser.add_argument('--height', type=int, default=500,
                       help='Image height (default: 500)')
    parser.add_argument('--point-radius', type=int, default=5,
                       help='Radius of point circles (default: 5)')
    parser.add_argument('--label-offset', type=int, default=20,
                       help='Distance of label from point (default: 20)')
    parser.add_argument('--label-font-size', type=int, default=14,
                       help='Font size for labels (default: 14)')

    args = parser.parse_args()

    create_labeled_tsps(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        show_tour=args.show_tour,
        label_offset=args.label_offset,
        label_font_size=args.label_font_size,
        format=args.format,
        width=args.width,
        height=args.height,
        point_radius=args.point_radius
    )


if __name__ == '__main__':
    main()
