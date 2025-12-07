#!/usr/bin/env python3
"""
Plot TSP points with random integer labels in SVG format.
Matches the exact style of the original TSP visualizations.
"""

import json
import random
import os
import math
from pathlib import Path


def find_label_position(point, all_points, offset_distance=20,
                       width=800, height=500, label_padding=15):
    """
    Find a good position for a label that avoids overlapping with nearby points
    and stays within image bounds.

    Args:
        point: (x, y) tuple of the point being labeled
        all_points: List of all (x, y) points
        offset_distance: Distance to offset the label from the point
        width: Image width for bounds checking
        height: Image height for bounds checking
        label_padding: Padding from image edges for label placement

    Returns:
        (x, y) position for label
    """
    x, y = point

    # Try 8 different positions around the point
    angles = [0, 45, 90, 135, 180, 225, 270, 315]

    # Evaluate each position
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
            'angle': angle,
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
    """
    Create SVG content with labeled points.

    Args:
        points: List of {"x": x, "y": y} dictionaries
        labels: List of integer labels (same length as points)
        width: SVG width
        height: SVG height
        point_radius: Radius of point circles
        label_offset: Distance of label from point (fixed distance, smart direction)
        label_font_size: Font size for labels
        show_tour: Whether to show the tour line
        tour_order: Optional list of indices for tour order

    Returns:
        SVG content as string
    """
    svg_parts = []

    # SVG header
    svg_parts.append(f'<svg width="{width}" xmlns="http://www.w3.org/2000/svg" height="{height}">')

    # White background
    svg_parts.append(f'  <rect width="{width}" fill="white" height="{height}" />')

    # Draw tour line if requested (before points so points appear on top)
    if show_tour and tour_order:
        path_points = [f"{points[i]['x']} {points[i]['y']}" for i in tour_order]
        path_d = "M " + " L ".join(path_points)
        svg_parts.append(f'  <path fill="none" stroke="blue" d="{path_d}" />')

    # Draw points (matching original style exactly)
    for p in points:
        svg_parts.append(f'  <circle cy="{p["y"]}" cx="{p["x"]}" r="{point_radius}" />')

    # Draw labels (no background, just text)
    all_point_coords = [(p['x'], p['y']) for p in points]

    for p, label in zip(points, labels):
        # Find good position for label (fixed distance, smart direction to avoid occlusion)
        label_x, label_y = find_label_position((p['x'], p['y']), all_point_coords,
                                               label_offset, width, height)

        # Draw label text only (no background)
        # text-anchor="middle" centers horizontally
        # dominant-baseline="central" centers vertically
        svg_parts.append(
            f'  <text x="{label_x:.1f}" y="{label_y:.1f}" '
            f'text-anchor="middle" dominant-baseline="central" '
            f'font-size="{label_font_size}" font-weight="bold" '
            f'font-family="Arial, sans-serif">{label}</text>'
        )

    # SVG footer
    svg_parts.append('</svg>')

    return '\n'.join(svg_parts)


def plot_tsp_with_random_labels(file_path, output_path=None,
                                width=800, height=500, seed=None,
                                point_radius=5, label_offset=20,
                                label_font_size=14,
                                show_tour=False):
    """
    Plot TSP points with random integer labels in SVG format.

    Args:
        file_path: Path to the TSP file containing JSON point data
        output_path: Path to save the SVG file
        width: SVG width (default: 800)
        height: SVG height (default: 500)
        seed: Random seed for reproducibility
        point_radius: Radius of point circles (default: 5)
        label_offset: Distance of label from point (default: 20)
        label_font_size: Font size for labels (default: 14)
        show_tour: Whether to show the tour line (default: False)
    """
    # Set random seed if provided
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

    # Save SVG and mapping
    if output_path:
        with open(output_path, 'w') as f:
            f.write(svg_content)
        print(f"Saved SVG to: {output_path}")

        # Save label mapping as JSON (for verification later)
        mapping_path = output_path.rsplit('.', 1)[0] + '_mapping.json'
        mapping = {
            'points': points,
            'labels': labels,
            'index_to_label': {i: label for i, label in enumerate(labels)},
            'label_to_index': {label: i for i, label in enumerate(labels)}
        }
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"Saved mapping to: {mapping_path}")
    else:
        print(svg_content)

    # Print mapping for reference
    print("\nPoint to Label Mapping:")
    print("Index | Coordinates | Label")
    print("-" * 40)
    for i, (p, label) in enumerate(zip(points, labels)):
        print(f"{i:5d} | ({p['x']:3d}, {p['y']:3d}) | {label:3d}")

    return labels


def main():
    """Main function to demonstrate usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot TSP points with random integer labels (SVG format)'
    )
    parser.add_argument('input_file',
                       help='Path to TSP file (JSON format)')
    parser.add_argument('-o', '--output',
                       help='Output file path for the SVG (e.g., plot.svg)')
    parser.add_argument('-s', '--seed', type=int,
                       help='Random seed for reproducibility')
    parser.add_argument('--width', type=int, default=800,
                       help='SVG width (default: 800)')
    parser.add_argument('--height', type=int, default=500,
                       help='SVG height (default: 500)')
    parser.add_argument('--point-radius', type=int, default=5,
                       help='Radius of point circles (default: 5)')
    parser.add_argument('--label-offset', type=int, default=20,
                       help='Distance of label from point (default: 20)')
    parser.add_argument('--label-font-size', type=int, default=14,
                       help='Font size for labels (default: 14)')
    parser.add_argument('--show-tour', action='store_true',
                       help='Show the tour line connecting points')

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' not found")
        return

    # Plot with random labels
    plot_tsp_with_random_labels(
        args.input_file,
        output_path=args.output,
        width=args.width,
        height=args.height,
        seed=args.seed,
        point_radius=args.point_radius,
        label_offset=args.label_offset,
        label_font_size=args.label_font_size,
        show_tour=args.show_tour
    )


if __name__ == '__main__':
    main()
