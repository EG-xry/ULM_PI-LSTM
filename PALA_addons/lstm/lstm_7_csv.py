import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(current_dir, 'mb_ulm_4con.csv')

try:
    # Read the CSV file
    df = pd.read_csv(csv_file)
    print(f"Successfully read file: {csv_file}")
    
    # Verify the required data columns
    required_columns = ['X', 'Y', 'ImageIndex']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV file is missing required columns. It must include: X, Y, ImageIndex")
    
    # Calculate the data range
    x_min, x_max = df['X'].min(), df['X'].max()
    y_min, y_max = df['Y'].min(), df['Y'].max()
    x_span = x_max - x_min
    y_span = y_max - y_min
    
    # Adjust coordinate margins
    x_min = df['X'].min() - 0.2 * x_span  # Add margin
    x_max = df['X'].max() + 0.2 * x_span
    y_min = df['Y'].min() - 0.2 * y_span
    y_max = df['Y'].max() + 0.2 * y_span
    
    # Modify size calculation parameters
    dpi = 150  # Increase DPI
    base_scale = 5  # Increase scaling factor
    
    # Calculate dimensions based on adjusted parameters
    fig_width = (y_span * base_scale) / dpi
    fig_height = (x_span * base_scale) / dpi
    
    # Ensure a minimum size
    min_size = 6  # Minimum 6 inches
    fig_width = max(fig_width, min_size)
    fig_height = max(fig_height, min_size)
    
    # Print dimension information
    print(f"Generated dimensions: {fig_width:.1f} x {fig_height:.1f} inches (DPI: {dpi})")
    print(f"Equivalent pixels: {int(fig_width * dpi)} x {int(fig_height * dpi)}")
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_subplot(111)
    
    # Initialize scatter plot with empty data
    scat = ax.scatter([], [],
                      s=1,
                      c=[],  # Initialize empty color array
                      cmap='Reds',
                      edgecolors='none',
                      norm=plt.Normalize(vmin=df['ImageIndex'].min(), 
                                         vmax=df['ImageIndex'].max()))
    
    # Initialize dictionary for trajectory lines (outside the animation function)
    traj_lines = {}
    
    # Configure animation parameters
    frame_step = max(1, len(df['ImageIndex'].unique()) // 100)  # Automatically compute frame step
    frames = sorted(df['ImageIndex'].unique()[::frame_step])
    
    # Set fixed coordinate ranges when initializing the canvas
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Update function for the animation
    def update(frame):
        frame_data = df[df['ImageIndex'] <= frame]
        
        # Set positions and colors for the scatter plot
        scat.set_offsets(frame_data[['X', 'Y']].values)
        scat.set_array(frame_data['ImageIndex'])  # Use ImageIndex to drive colors
        
        # Dynamically adjust color range
        scat.norm.vmax = frame  # Latest frame appears in most red
        
        ax.set_title(f"Frame: {frame}/{frames[-1]} ({len(frame_data)} points)")
        return (scat,) + tuple(traj_lines.values())  # Ensure all graphic elements are returned
    
    # Create and save the animation
    ani = animation.FuncAnimation(
        fig, update, 
        frames=frames,
        interval=50, 
        blit=True
    )
    
    # Add a timestamp to generate a unique file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(current_dir, f'ulm_animation_{timestamp}.gif')
    
    # Warn if the file exists; will be overwritten
    if os.path.exists(output_file):
        print(f"Warning: Overwriting existing file {output_file}")
    
    ani.save(output_file, writer='pillow', dpi=150)
    print(f"Animation saved to: {output_file}")

except FileNotFoundError:
    print(f"Error: File {csv_file} not found")
except pd.errors.EmptyDataError:
    print("Error: CSV file is empty or has an incorrect format")
except Exception as e:
    print(f"An error occurred during processing: {str(e)}")
