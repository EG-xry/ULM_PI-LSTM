import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def load_mat_data(file_path):
    """Load MATLAB .mat file"""
    try:
        data = loadmat(file_path)
        print("Variables contained in the file:")
        for key in data.keys():
            if not key.startswith('__'):
                print(f"- {key}: {type(data[key])} {data[key].shape}")
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} does not exist")
        return None
    except Exception as e:
        print(f"Parsing error: {str(e)}")
        return None

def parse_ultrasound_data(mat_data):
    """Parse MATLAB ultrasound data structure"""
    # Check for required keys
    required_keys = ['fs', 'rf', 'iq']
    missing = [key for key in required_keys if key not in mat_data]
    if missing:
        raise KeyError(f"MAT data is missing required fields: {missing}")

    # Get sampling frequency
    fs = mat_data['fs'].item() if isinstance(mat_data['fs'], np.ndarray) else mat_data['fs']
    
    # Automatically generate time axis if 't' field is missing
    if 't' not in mat_data:
        num_samples = mat_data['rf'].shape[0]
        t = np.arange(num_samples) / fs
        print("Warning: Time axis automatically generated")
    else:
        t = mat_data['t'].flatten()
    
    rf = mat_data['rf']  # RF signal
    iq = mat_data['iq']  # IQ data
    
    # Verify dimensions
    assert rf.shape[0] == len(t), "RF data does not match time axis"
    print(f"Data loaded successfully! Sampling frequency: {fs/1e6:.2f} MHz")

    # Optional: Process depth field (default to 0)
    depth = mat_data.get('depth', 0)
    
    return {
        'fs': fs,
        't': t,
        'rf': rf,
        'iq': iq,
        'depth': depth  # Add depth field
    }

def visualize_iq(data_dict, scan_line=0):
    """Visualize IQ signals"""
    iq = data_dict['iq']
    t = np.arange(iq.shape[1]) / data_dict['fs'] * 1e6  # Convert to microseconds
    
    plt.figure(figsize=(12, 6))
    
    # Raw IQ signals
    plt.subplot(1, 2, 1)
    plt.plot(t, iq[scan_line, :, 0], label='I Channel')
    plt.plot(t, iq[scan_line, :, 1], label='Q Channel')
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.title(f'Raw IQ Signals (Scan Line {scan_line})')
    plt.legend()
    
    # Envelope signal
    envelope = np.abs(iq[scan_line, :, 0] + 1j*iq[scan_line, :, 1])
    plt.subplot(1, 2, 2)
    plt.plot(t, envelope)
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.title('Envelope Detection')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    mat_file = "ultrasound_data.mat"
    
    # 1. Load data
    mat_data = load_mat_data(mat_file)
    if mat_data is None:
        # Generate sample data for testing
        print("\nGenerating sample data...")
        example_data = {
            'rf': np.random.randn(1000, 64),  # Add rf data
            'iq': np.random.randn(128, 1000, 2).astype(np.float32),  # Fix key name for IQ data
            'fs': 40e6,
            'depth': 0.08,
            'comment': 'Example ultrasound data'
        }
        ultrasound_data = parse_ultrasound_data(example_data)
    else:
        # 2. Parse data
        ultrasound_data = parse_ultrasound_data(mat_data)
    
    # 3. Display basic information
    print("\nParsed data structure:")
    print(f"- IQ data shape: {ultrasound_data['iq'].shape}")
    print(f"- Sampling frequency: {ultrasound_data['fs']/1e6:.1f} MHz")
    
    # Check if the depth field exists
    if 'depth' in ultrasound_data and ultrasound_data['depth'] is not None:
        print(f"- Depth: {ultrasound_data['depth']*100:.1f} cm")
    else:
        print("- Depth: Not provided")
    
    # 4. Visualize
    visualize_iq(ultrasound_data)
