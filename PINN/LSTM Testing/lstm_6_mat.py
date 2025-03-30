import numpy as np
from scipy.io import loadmat
import os
import pandas as pd
import h5py
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import stats

def print_mat_structure(file_path):
    """Recursively print the structure of the MATLAB file"""
    try:
        full_path = os.path.abspath(file_path)
        print(f"Current working directory: {os.getcwd()}")
        print(f"Attempting to load path: {full_path}")
        
        if not os.path.exists(full_path):
            dir_path = os.path.dirname(full_path)
            mat_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.mat')]
            print("\nMAT files found in directory:")
            print('\n'.join(mat_files) or "No .mat files found")
            raise FileNotFoundError(f"File not found: {full_path}")
            
        data = loadmat(full_path)
        print(f"\nSuccessfully loaded file: {os.path.basename(full_path)}")
        print("="*50)
        
        def print_recursive(d, indent=0):
            space = "  " * indent
            for key in d.keys():
                if key.startswith('__'):
                    continue
                item = d[key]
                type_str = str(type(item)).split("'")[1]
                shape_str = f"shape={item.shape}" if hasattr(item, 'shape') else ""
                print(f"{space}├─ {key} ({type_str}) {shape_str}")
                
                if isinstance(item, np.ndarray) and item.dtype.names:
                    print(f"{space}│  └─ Struct fields: {item.dtype.names}")
                    for field in item.dtype.names:
                        field_sample = item[0][0][field] if item.size == 1 else item[0][0][field][0]
                        print(f"{space}│     ├─ {field}: {type(field_sample)} {field_sample.shape if hasattr(field_sample, 'shape') else ''}")
                elif isinstance(item, np.ndarray) and item.dtype == object:
                    if item.size > 0:
                        cell_sample = item.flat[0]
                        print(f"{space}│  └─ Cell content example: {type(cell_sample)} {cell_sample.shape if hasattr(cell_sample, 'shape') else ''}")
        
        print_recursive(data)
        
    except Exception as e:
        print(f"\nError details: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check if the file name matches exactly (including case)")
        print("2. Verify the file extension (it is recommended to include the file extension)")
        print("3. Try using an absolute path:")
        print(f"   print_mat_structure(r'C:\\full\\path\\PALA_InVivoRatBrain__tracks001.mat')")
        print("4. Check file permissions (right-click properties - security settings)")

def save_to_csv(data, mat_path, var_name):
    """Save the specified variable to a CSV file"""
    try:
        base_name = os.path.splitext(mat_path)[0]
        csv_path = f"{base_name}_{var_name}.csv"
        df = pd.DataFrame(data, columns=['X', 'Y'])
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Successfully saved: {csv_path}")
        print(f"Data dimensions: {data.shape} -> Saved records: {len(df)}")
    except Exception as e:
        print(f"Failed to save CSV: {str(e)}")

def extract_ulm_info(mat_data):
    """Extract ULM structure information"""
    if 'Track_tot_i' not in mat_data:
        raise KeyError("The 'tracks' structure was not found in the MAT file. Please verify that it is a ULM data file")
    
    tracks = mat_data['Track_tot_i'][0]
    ulm_info = []
    
    for track_id, track in enumerate(tracks):
        pos = track['pos'][0][0]
        t = track['t'][0][0].flatten()
        amp = track['amp'][0][0][0][0]
        vel = track['vel'][0][0][0][0] if 'vel' in track.dtype.names else np.nan
        
        for i in range(len(t)):
            ulm_info.append({
                'TrackID': track_id + 1,
                'TimePoint': i + 1,
                'X': pos[i, 0],
                'Y': pos[i, 1],
                'Time': t[i],
                'Amplitude': amp,
                'Velocity': vel
            })
    return pd.DataFrame(ulm_info)

def save_ulm_metadata(mat_data, csv_path):
    """Save ULM acquisition parameters"""
    meta = []
    if 'parameters' in mat_data:
        params = mat_data['parameters'][0][0]
        meta.append({'Parameter': 'Sampling Frequency', 'Value': params['fs'][0][0], 'Unit': 'Hz'})
        meta.append({'Parameter': 'Depth', 'Value': params['depth'][0][0], 'Unit': 'm'})
        meta.append({'Parameter': 'Wavelength', 'Value': params['wavelength'][0][0], 'Unit': 'm'})
    
    if 'acquisition_date' in mat_data:
        date_str = ''.join(mat_data['acquisition_date'][0])
        meta.append({'Parameter': 'Acquisition Date', 'Value': date_str, 'Unit': ''})
    
    pd.DataFrame(meta).to_csv(csv_path, index=False)

def save_ulm_structure(mat_data, save_path):
    """Save complete ULM structure data"""
    try:
        with h5py.File(save_path, 'w') as hf:
            # Safely access MATLAB cell array
            tracks = np.squeeze(mat_data['Track_tot_i'])
            track_group = hf.create_group('tracks')
            
            for i, track in enumerate(tracks):
                # Safely access track data
                try:
                    pos_data = np.array(track['pos'].item(), dtype=np.float32)
                    t_data = np.array(track['t'].item().flatten(), dtype=np.float32)
                    amp_value = float(np.array(track['amp'].item()).item())
                except (AttributeError, IndexError) as e:
                    print(f"Warning: Track {i+1} data structure abnormal ({str(e)}), skipping")
                    continue
                
                # Create track group
                track_grp = track_group.create_group(f'track_{i+1}')
                track_grp.create_dataset('pos', data=pos_data)
                track_grp.create_dataset('t', data=t_data)
                track_grp.attrs['amplitude'] = amp_value
                
                # Handle optional fields
                if 'vel' in track.dtype.names:
                    try:
                        vel_value = float(np.array(track['vel'].item()).item())
                        track_grp.attrs['velocity'] = vel_value
                    except:
                        track_grp.attrs['velocity'] = np.nan

            # Process parameters
            if 'parameters' in mat_data:
                param_grp = hf.create_group('parameters')
                params = mat_data['parameters'].item()
                
                # Safely retrieve parameters
                param_mapping = {
                    'fs': ('Sampling Frequency', 'Hz'),
                    'depth': ('Depth', 'm'),
                    'wavelength': ('Wavelength', 'm')
                }
                
                for field in param_mapping:
                    try:
                        value = float(np.array(params[field].item()).item())
                        param_grp.attrs[field] = value
                    except:
                        param_grp.attrs[field] = 0.0
                        print(f"Parameter {field} retrieval failed, default value 0 set")

            hf.attrs['export_date'] = datetime.now().isoformat()
        print(f"ULM structure has been saved to: {save_path}")
    except Exception as e:
        print(f"Failed to save: {str(e)}")

def detect_outliers(data, threshold=3):
    """Detect outliers"""
    z_scores = stats.zscore(data)
    return np.abs(z_scores) > threshold

def visualize_data(ulm_df, tracking_data):
    """Visualize ULM tracks and tracking points"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    for track_id in ulm_df['TrackID'].unique()[:10]:
        track = ulm_df[ulm_df['TrackID'] == track_id]
        plt.plot(track['X'], track['Y'], marker='o', markersize=3, 
                linestyle='-', linewidth=0.5, label=f'Track {track_id}')
    plt.title('ULM Microbubble Trajectories')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.gca().invert_yaxis()
    
    plt.subplot(1, 2, 2)
    plt.scatter(tracking_data[:,0], tracking_data[:,1], s=1, alpha=0.3, c='red')
    plt.title('Tracking Points Distribution')
    plt.xlabel('X Position (pixels)')
    plt.gca().invert_yaxis()
    
    plt.scatter(ulm_df['X'], ulm_df['Y'], c=ulm_df['Time'], 
               cmap='viridis', s=5, alpha=0.5)
    plt.colorbar(label='Time (s)')
    
    scale_length = 200
    plt.plot([50, 50+scale_length], [50, 50], 'k-', linewidth=2)
    plt.text(50+scale_length/2, 55, '10 mm', ha='center')
    
    plt.xlim(ulm_df['X'].min()-10, ulm_df['X'].max()+10)
    plt.ylim(ulm_df['Y'].max()+10, ulm_df['Y'].min()-10)
    
    plt.tight_layout()
    img_path = os.path.splitext(mat_file)[0] + "_visualization.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight', 
               facecolor='white', transparent=False)
    print(f"\nVisualization result has been saved to: {img_path}")
    plt.show()

def load_mat_data(file_path):
    """Safely load the MAT file"""
    try:
        return loadmat(file_path, 
                      struct_as_record=False,
                      squeeze_me=True,  # Automatically squeeze singleton arrays
                      mat_dtype=True,
                      simplify_cells=True)  # Automatically simplify cell structures
    except Exception as e:
        print(f"MAT file loading failed: {str(e)}")
        return None

if __name__ == "__main__":
    mat_file = "PALA_InVivoRatBrain__tracks001.mat"
    safe_mat_file = mat_file.replace(' ', '_')
    print(f"Sanitized file name: {safe_mat_file}")
    
    print_mat_structure(safe_mat_file)
    
    try:
        data = load_mat_data(safe_mat_file)
        
        # Process MatTracking_Pixel
        if 'MatTracking_Pixel' in data:
            tracking_data = data['MatTracking_Pixel']
            print(f"\nTracking data statistics:")
            print(f"- X range: [{tracking_data[:,0].min():.2f}, {tracking_data[:,0].max():.2f}] pixels")
            print(f"- Y range: [{tracking_data[:,1].min():.2f}, {tracking_data[:,1].max():.2f}] pixels")
            
            csv_path = os.path.splitext(mat_file)[0] + "_MatTracking_Pixel.csv"
            pd.DataFrame(tracking_data, columns=['X', 'Y']).to_csv(csv_path, index=False)
            print(f"Coordinate data has been saved to: {csv_path}")
            
            outliers = detect_outliers(tracking_data)
            print(f"Number of outliers found: {np.sum(outliers.any(axis=1))}")
        
        # Process ULM data
        if 'Track_tot_i' in data:
            ulm_h5_path = os.path.splitext(mat_file)[0] + "_ULM.h5"
            save_ulm_structure(data, ulm_h5_path)
            
            df_ulm = extract_ulm_info(data)
            csv_path = os.path.splitext(mat_file)[0] + "_ULM_tracks.csv"
            df_ulm.to_csv(csv_path, index=False)
            print(f"Track table has been saved to: {csv_path}")
            
            meta_csv = os.path.splitext(mat_file)[0] + "_ulmmeta.csv"
            save_ulm_metadata(data, meta_csv)
            print(f"Acquisition parameters have been saved to: {meta_csv}")
            
            if 'MatTracking_Pixel' in data:
                visualize_data(df_ulm, tracking_data)
                
                # Create animation
                fig, ax = plt.subplots()
                scat = ax.scatter([], [], s=5)
                ax.invert_yaxis()

                def animate(frame):
                    frame_data = df_ulm[df_ulm['Time'] <= frame/10]
                    scat.set_offsets(frame_data[['X', 'Y']])
                    return scat,

                ani = FuncAnimation(fig, animate, frames=100, interval=50)
                ani.save('trajectories.gif', writer='pillow')
                print("Track animation has been saved as trajectories.gif")
                
    except Exception as e:
        print(f"Runtime error: {str(e)}")
