#!/usr/bin/env python3
"""
Plot TSP points with random integer labels in PNG format.
Matches the exact style of the original TSP visualizations but outputs raster images.
"""

import json
import random
import os
import math
from pathlib import Path

# NEW: Import Pillow library
from PIL import Image, ImageDraw, ImageFont

def find_label_position(point, all_points, offset_distance=20,
                        width=800, height=500, label_padding=15):
    """
    Find a good position for a label that avoids overlapping with nearby points
    and stays within image bounds.
    (Logic unchanged from original script)
    """
    x, y = point

    # Try 8 different positions around the point
    angles = [0, 45, 90, 135, 180, 225, 270, 315]

    candidates = []

    for angle in angles:
        rad = math.radians(angle)
        label_x = x + offset_distance * math.cos(rad)
        label_y = y + offset_distance * math.sin(rad)

        in_bounds = (label_padding <= label_x <= width - label_padding and
                     label_padding <= label_y <= height - label_padding)

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

    in_bounds_candidates = [c for c in candidates if c['in_bounds']]

    if in_bounds_candidates:
        best = max(in_bounds_candidates, key=lambda c: c['min_dist'])
    else:
        best = max(candidates, key=lambda c: c['min_dist'])
        best['x'] = max(label_padding, min(width - label_padding, best['x']))
        best['y'] = max(label_padding, min(height - label_padding, best['y']))

    return best['x'], best['y']


def create_png_image(points, labels, width=800, height=500,
                          point_radius=5, label_offset=20,
                          label_font_size=14,
                          show_tour=False, tour_order=None):
    """
    Create a PIL Image object with labeled points.
    """
    # 1. Create white background image
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    # 2. Load Font
    # Try to load Arial, fall back to default if not found
    try:
        font = ImageFont.truetype("arial.ttf", label_font_size)
    except IOError:
        # Fallback for Linux/Mac or if arial is missing
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", label_font_size)
        except IOError:
            print("Warning: Custom fonts not found, using default (size may be small)")
            font = ImageFont.load_default()

    # 3. Draw tour line if requested (Draw first so it's under points)
    if show_tour and tour_order:
        tour_coords = [(points[i]['x'], points[i]['y']) for i in tour_order]
        # Draw line connecting points in order
        draw.line(tour_coords, fill="blue", width=2)

    # 4. Draw points
    for p in points:
        # PIL draws ellipses using a bounding box [x0, y0, x1, y1]
        x, y = p['x'], p['y']
        bbox = [x - point_radius, y - point_radius, x + point_radius, y + point_radius]
        draw.ellipse(bbox, fill="black", outline=None)

    # 5. Draw labels
    all_point_coords = [(p['x'], p['y']) for p in points]

    for p, label in zip(points, labels):
        # Use existing logic to find position
        label_x, label_y = find_label_position((p['x'], p['y']), all_point_coords,
                                               label_offset, width, height)
        
        label_text = str(label)
        
        # Calculate text size to center it (SVG does this automatically, PIL does not)
        # using textbbox (left, top, right, bottom)
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Center the text at label_x, label_y
        draw_x = label_x - (text_width / 2)
        draw_y = label_y - (text_height / 2)

        draw.text((draw_x, draw_y), label_text, fill="black", font=font)

    return image


def plot_tsp_with_random_labels(file_path, output_path=None,
                                width=800, height=500, seed=None,
                                point_radius=5, label_offset=20,
                                label_font_size=14,
                                show_tour=False):
    """
    Main orchestration function.
    """
    if seed is not None:
        random.seed(seed)

    with open(file_path, 'r') as f:
        points = json.load(f)

    n_points = len(points)
    labels = list(range(1, n_points + 1))
    random.shuffle(labels)

    tour_order = list(range(n_points)) if show_tour else None

    # Create the PIL Image
    img = create_png_image(
        points, labels, width, height,
        point_radius, label_offset,
        label_font_size,
        show_tour, tour_order
    )

    # Save Image
    if output_path:
        # Ensure extension is .png
        if not output_path.lower().endswith('.png'):
            output_path += '.png'
            
        img.save(output_path, "PNG")
        print(f"Saved PNG to: {output_path}")

        # Save mapping (Same as before)
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
        # If no output path, just show it (useful for debugging)
        img.show()

    return labels


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot TSP points with random integer labels (PNG format)'
    )
    parser.add_argument('input_file',
                        help='Path to TSP file (JSON format)')
    parser.add_argument('-o', '--output',
                        help='Output file path for the PNG (e.g., plot.png)')
    parser.add_argument('-s', '--seed', type=int,
                        help='Random seed for reproducibility')
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
    parser.add_argument('--show-tour', action='store_true',
                        help='Show the tour line connecting points')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' not found")
        return

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