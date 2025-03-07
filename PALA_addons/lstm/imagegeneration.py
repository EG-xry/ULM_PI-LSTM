import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
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
    
    # Loop through each track id and plot the corresponding points and line.
    for track in track_ids:
        # Select data for the current track and sort by point_index.
        df_track = df[df['track_id'] == track].sort_values(by='point_index')
        x_points = df_track['x']
        y_points = df_track['y']
        
        # Plot the track with a thin red line and small red markers.
        plt.plot(x_points, y_points, color='red', linewidth=0.5,
                 marker='o', markersize=3, markerfacecolor='red')
    
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

if __name__ == "__main__":
    main()