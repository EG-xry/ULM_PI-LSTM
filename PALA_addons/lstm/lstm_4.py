import numpy as np
import h5py
from scipy.io import loadmat
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_iq_data(file_path):
    """Load ultrasound IQ data (supports .mat and .h5 formats)"""
    if file_path.endswith('.mat'):
        data = loadmat(file_path)
        iq_data = data['iq_data']  # assuming data is stored in the 'iq_data' variable
    elif file_path.endswith('.h5'):
        with h5py.File(file_path, 'r') as f:
            iq_data = f['iq_data'][:]
    else:
        raise ValueError("Unsupported file format")
    
    print(f"Original data shape: {iq_data.shape} (scan_lines, samples)")
    return iq_data


def preprocess_iq(iq_data, window_size=64, stride=32):
    """Preprocess IQ data into a format suitable for deep learning"""
    # Convert to complex data
    iq_complex = iq_data[:, :, 0] + 1j * iq_data[:, :, 1]
    
    # Compute envelope (magnitude)
    envelope = np.abs(iq_complex)
    
    # Log compression (dB)
    envelope_db = 20 * np.log10(envelope + 1e-6)
    
    # Normalize to 0-1 range
    envelope_norm = (envelope_db - np.min(envelope_db)) / (np.max(envelope_db) - np.min(envelope_db))
    
    # Create sliding window samples
    samples = []
    for line in envelope_norm:
        for i in range(0, len(line)-window_size, stride):
            samples.append(line[i:i+window_size])
    
    return np.array(samples)

def create_keras_dataset(data, labels, batch_size=32):
    """Create a TensorFlow data pipeline"""
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Example usage
if __name__ == "__main__":
    # 1. Try to load data
    file_path = "path/to/your/ultrasound_iq.h5"
    try:
        iq_data = load_iq_data(file_path)
    except FileNotFoundError:
        print(f"Error: file {file_path} not found")
        print("Possible solutions:")
        print("1. Ensure the file exists in the current working directory")
        print("2. Alternatively, provide an absolute path to the file")
        print("3. Generating sample data for testing...")
        
        # Generate sample data
        num_scan_lines = 100
        num_samples = 500
        iq_data = np.random.randn(num_scan_lines, num_samples, 2).astype(np.float32)
        print(f"Generated sample data: {iq_data.shape}")
    
    # 2. Data preprocessing
    processed_data = preprocess_iq(iq_data)
    print(f"Processed data shape: {processed_data.shape} (samples, window_size)")
    
    # 3. Add channel dimension (for CNN)
    processed_data = processed_data[..., np.newaxis]  # new shape: (samples, window_size, 1)
    
    # 4. Create labels (example: binary classification)
    labels = np.random.randint(0, 2, size=len(processed_data))  # replace with actual labels
    
    # 5. Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        processed_data, labels, 
        test_size=0.2, 
        random_state=42
    )
    
    # 6. Create data pipeline
    train_dataset = create_keras_dataset(X_train, y_train)
    val_dataset = create_keras_dataset(X_val, y_val)
    
    # 7. Example model input
    sample_input = X_train[0]
    print(f"Single sample input shape: {sample_input.shape} (window_size, channels)")
