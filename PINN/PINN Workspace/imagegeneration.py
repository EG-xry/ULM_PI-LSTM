import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import matplotlib.animation as animation
import numpy as np

def main(small_points=True):
    # Build the full path for the CSV file relative to the script's directory.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    csv_path = os.path.join(script_dir, 'tracks.csv')

    # Load tracking data from CSV file.
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file 'tracks.csv' was not found at {csv_path}")

    # Verify that the CSV has all the required columns.
    required_columns = ['track_id', 'point_index', 'x', 'y', 't']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("The CSV file does not contain all the required columns: track_id, point_index, x, y, t")

    # Get unique track IDs.
    track_ids = df['track_id'].unique()
    
    # Create the plot with a white background.
    plt.figure(figsize=(10, 8), facecolor='w')
    
    # Toggle marker size: use a smaller marker if small_points is True (now default True).
    marker_size = 1 if small_points else 3
    
    # Loop through each track id and plot the corresponding points and line.
    for track in track_ids:
        # Select data for the current track and sort by point_index.
        df_track = df[df['track_id'] == track].sort_values(by='point_index')
        x_points = df_track['x']
        y_points = df_track['y']
        
        # Plot the track with a thin red line and markers.
        plt.plot(x_points, y_points, color='red', linewidth=0.5,
                 marker='o', markersize=marker_size, markerfacecolor='red')
    
    # Label the axes and add a title.
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Plain Brain Map of Tracks")
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()

    # Save the figure as PNG and TIFF.
    plt.savefig("PlainBrainMap.png", dpi=300)
    plt.savefig("PlainBrainMap.tif", dpi=300)

    # Display the plot.
    plt.show()

def create_animation(fast=True):
    # Build the full path for the CSV file relative to the script's directory.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    csv_path = os.path.join(script_dir, 'tracks.csv')

    # Load tracking data from CSV file.
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file 'tracks.csv' was not found at {csv_path}")
    
    # Verify that the CSV has all the required columns.
    required_columns = ['track_id', 'point_index', 'x', 'y', 't']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("The CSV file does not contain all the required columns: track_id, point_index, x, y, t")
    
    # Get unique frame numbers. We assume column 't' holds the frame number.
    frames = sorted(df['t'].unique())

    # Determine x and y limits so that the axes remain constant for all frames.
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()

    # Group the data by track id and store the ordered positions and corresponding frames.
    track_groups = {}
    for track_id, group in df.groupby('track_id'):
        group_sorted = group.sort_values(by='t')
        track_groups[track_id] = {
            't': group_sorted['t'].values,
            'x': group_sorted['x'].values,
            'y': group_sorted['y'].values
        }

    # Create the matplotlib figure and axis.
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    title = ax.set_title("Frame 1")

    # Pre-create a Line2D object for each track to display its trail.
    lines = {}
    for track_id in track_groups.keys():
        # Create an initially empty line; this will update as the track progresses.
        line, = ax.plot([], [], color='red', linewidth=0.5)
        lines[track_id] = line

    # Create a scatter to hold the current microbubble positions.
    dot_scatter = ax.scatter([], [], c='red', s=20)

    def init():
        # Initialize all line data and the scatter.
        for line in lines.values():
            line.set_data([], [])
        dot_scatter.set_offsets(np.empty((0, 2)))
        return list(lines.values()) + [dot_scatter, title]

    def update(frame):
        current_dots = []
        # For each track, update its trail from the start until the current frame.
        for track_id, data in track_groups.items():
            # Find indices where the track's frame number is less than or equal to the current frame.
            indices = np.where(data['t'] <= frame)[0]
            if len(indices) > 0:
                x_vals = data['x'][indices]
                y_vals = data['y'][indices]
                lines[track_id].set_data(x_vals, y_vals)
                # If the track has a position exactly at the current frame, collect that for the current dot.
                current_idx = np.where(data['t'] == frame)[0]
                if len(current_idx) > 0:
                    # Add all points corresponding to the current frame. (There may be more than one per track.)
                    for i in current_idx:
                        current_dots.append([data['x'][i], data['y'][i]])
            else:
                lines[track_id].set_data([], [])
        # Update the red dots representing the current microbubble positions.
        if current_dots:
            dot_scatter.set_offsets(np.array(current_dots))
        else:
            dot_scatter.set_offsets(np.empty((0, 2)))
        title.set_text(f"Frame {frame}")
        return list(lines.values()) + [dot_scatter, title]
    
    # Set the interval (in ms) based on the 'fast' flag.
    interval = 20 if fast else 500
    ani = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init,
        interval=interval, blit=False, repeat=False
    )

    # Save the animation as a GIF using Pillow.
    # The fps is set higher for faster animation.
    gif_fps = 2 if not fast else 5
    gif_path = os.path.join(script_dir, "microbubblePINN_animation.gif")
    ani.save(gif_path, writer='pillow', fps=gif_fps)
    print(f"GIF saved to {gif_path}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a plain brain map of tracks and a microbubble animation. " +
                    "Use --no-small-points to switch to larger markers, and --fast-animation " +
                    "to speed up the animation."
    )
    # Default is to use smaller points (small_points=True).
    parser.add_argument("--no-small-points", dest="small_points", action="store_false",
                        help="Toggle to use larger point markers (default is small)", default=True)
    parser.add_argument("--fast-animation", dest="fast_animation", action="store_true",
                        help="Enable faster frame progression in the animation", default=False)
    args = parser.parse_args()
    main(small_points=args.small_points)
    create_animation(fast=args.fast_animation)