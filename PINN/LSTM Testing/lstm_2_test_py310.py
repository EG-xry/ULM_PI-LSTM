"""
Complete workflow for time series forecasting using LSTM.
Features include: model caching, performance evaluation, visualization, and cross-period prediction.
Added to a private GitHub repository.
"""
# === Environment Setup ===
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
from tensorflow import GradientTape
import tensorflow as tf
from scipy.signal import welch

# Resolve OpenMP conflicts and adjust font settings for plotting
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
plt.rcParams['font.sans-serif'] = ['SimHei']  # Optional: set Chinese font for Windows (if needed)
plt.rcParams['axes.unicode_minus'] = False    # Ensure minus sign displays correctly

# === Data Preparation ===
def generate_synthetic_data(timesteps=1000):
    """Generate synthetic data containing multiple physical quantities."""
    t = np.linspace(0, 10, timesteps)
    # Base blood velocity signal
    velocity = 0.3 * np.sin(2 * np.pi * t / 0.8) + 0.1 * np.sin(4 * np.pi * t / 0.8 + np.pi / 3)
    # Compute related physical quantities
    pressure = 80 + 10 * velocity  # Blood pressure (mmHg)
    diameter = 4.0 + 0.5 * np.sin(velocity)  # Vessel diameter (mm)
    return np.column_stack([velocity, pressure, diameter])

def create_sequences(data, lookback=20):
    """Create multi-target supervised sequences."""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback, 0])  # Use only velocity for input
        y.append(data[i + lookback, :])    # Output includes all physical quantities
    return np.array(X), np.array(y)

# Main Data Generation
raw_data = generate_synthetic_data()
X_sequences, y_target = create_sequences(raw_data)

# === Data Normalization ===
# Standardize the data for each feature independently
data_mean = np.mean(y_target, axis=0)  # [velocity_mean, pressure_mean, diameter_mean]
data_std = np.std(y_target, axis=0)    # Corresponding standard deviations

# Normalize the input using the velocity's statistics only
X_normalized = (X_sequences - data_mean[0]) / data_std[0]
y_normalized = (y_target - data_mean) / data_std
X_normalized = X_normalized.reshape(-1, 20, 1)  # Reshape for LSTM input

# === Additional Configuration Options ===

# Training flag (users can modify; set to False to load a pre-trained model)
TRAIN_MODEL = True  

# Generate model and parameters file names using a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_PATH = f'lstm_model_{timestamp}.h5' if TRAIN_MODEL else None
PARAMS_PATH = f'model_params_{timestamp}.npz' if TRAIN_MODEL else None

# === Model Loading Logic ===
if not TRAIN_MODEL:
    # Find the latest model file in the directory
    model_files = [f for f in os.listdir() if f.startswith('lstm_model_')]
    if model_files:
        # Sort by timestamp and select the latest model
        latest_model = sorted(model_files)[-1]
        MODEL_PATH = latest_model
        PARAMS_PATH = latest_model.replace('lstm_model_', 'model_params_').replace('.h5', '.npz')
        print(f"⏳ Loading latest model: {MODEL_PATH}")
        # Load the model with custom loss objects
        model = load_model(MODEL_PATH, custom_objects={
            'physics_informed_loss': physics_informed_loss,
            'physics_constraint': physics_constraint
        })
        params = np.load(PARAMS_PATH)
        data_mean, data_std = params['mean'], params['std']
    else:
        raise FileNotFoundError("No model files found.")

# === Physics Constraint Module ===
def physics_constraint(y_true, y_pred, data_std):
    """
    Physics constraints specific for cerebral blood flow.
    Includes:
    1. Continuity equation (mass conservation)
    2. Simplified Navier-Stokes constraints (momentum conservation)
    3. Vessel wall shear stress constraint (τ = 4μv/D)
    4. Periodicity of pulsatile flow
    """
    # Revert standardization for physical quantities (assume y_pred represents blood velocity in m/s)
    velocity = y_pred * data_std
    
    with GradientTape(persistent=True) as g:
        g.watch(velocity)
        # Compute spatial derivative (assuming the input sequence represents spatial distribution)
        dv_dx = g.gradient(velocity, velocity)
        # Vessel diameter parameter (example value in mm)
        D = 4.0 + 0.5 * tf.sin(velocity)
        # Compute flow rate: Q = π*(D/2)^2 * v (converted to metric units)
        Q = np.pi * tf.square(D / 2000) * velocity
    # Constraint 1: Continuity equation ∂Q/∂x = -∂A/∂t (simplified version)
    continuity_loss = tf.reduce_mean(tf.square(dv_dx + 0.1 * velocity))
    
    # Constraint 2: Momentum conservation (simplified Navier-Stokes)
    with GradientTape() as g2:
        g2.watch(velocity)
        pressure = 80 + 10 * velocity  # Simulate relationship between blood pressure and flow velocity (mmHg)
    dp_dx = g2.gradient(pressure, velocity)
    momentum_loss = tf.reduce_mean(tf.square(1.05 * dv_dx + dp_dx))
    
    # Constraint 3: Vessel wall shear stress constraint (τ = 4μv/D)
    mu = 0.0035  # Blood viscosity (Pa·s)
    tau = 4 * mu * velocity / (D / 1000)  # Convert to metric units
    shear_loss = tf.reduce_mean(tf.square(tau - 2.5))  # Normal range is 2-4 Pa
    
    # Constraint 4: Periodicity of pulsatile flow
    period = 0.8  # Assumed period (~75 beats per minute)
    phase = tf.sin(2 * np.pi * velocity / period)
    periodic_loss = tf.reduce_mean(tf.square(phase))
    
    return 0.2 * continuity_loss + 0.3 * momentum_loss + 0.1 * shear_loss + 0.05 * periodic_loss

# === Model Building Function ===
def build_multi_task_model(input_shape):
    """
    Build a multi-task learning model.
    Outputs:
    - Blood velocity prediction
    - Blood pressure prediction
    - Vessel diameter prediction
    """
    input_layer = tf.keras.Input(shape=input_shape)
    
    # Shared feature extraction layers
    x = Conv1D(64, 3, activation='relu')(input_layer)
    x = MaxPooling1D(2)(x)
    x = LSTM(64, return_sequences=False)(x)
    
    # Multi-task output branches
    velocity = Dense(1, name='velocity')(x)
    pressure = Dense(1, name='pressure')(x)
    diameter = Dense(1, name='diameter')(x)
    
    model = tf.keras.Model(inputs=input_layer, outputs=[velocity, pressure, diameter])
    
    loss_weights = {'velocity': 0.6, 'pressure': 0.3, 'diameter': 0.1}
    
    model.compile(
        optimizer='adam',
        loss={
            'velocity': physics_informed_loss(data_std),
            'pressure': 'mse',
            'diameter': 'mae'
        },
        loss_weights=loss_weights,
        metrics={'velocity': ['mae'], 'pressure': ['mae'], 'diameter': ['mae']}
    )
    return model

# === Composite Loss Function ===
def physics_informed_loss(data_std):
    def combined_loss(y_true, y_pred):
        mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        physics_loss = physics_constraint(y_true, y_pred, data_std)
        return mse_loss + physics_loss
    return combined_loss

# === Model Training Logic ===
if TRAIN_MODEL or (not os.path.exists(MODEL_PATH)) or (not os.path.exists(PARAMS_PATH)):
    print("⏳ Start training new model...")
    model = build_multi_task_model((20, 1))
    X_multi = X_normalized.reshape(-1, 20, 1)
    y_multi = {
        'velocity': (y_target[:, 0] - data_mean[0]) / data_std[0],
        'pressure': (y_target[:, 1] - data_mean[1]) / data_std[1],
        'diameter': (y_target[:, 2] - data_mean[2]) / data_std[2]
    }
    
    start_time = time.time()
    training_history = model.fit(
        X_multi, y_multi,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    train_duration = time.time() - start_time
    
    model.save(MODEL_PATH)
    np.savez(PARAMS_PATH, mean=data_mean, std=data_std)
    print(f"💾 New model saved:\n- {MODEL_PATH}\n- {PARAMS_PATH}")
else:
    model = load_model(MODEL_PATH)
    params = np.load(PARAMS_PATH)
    data_mean, data_std = params['mean'], params['std']
    print("✅ Using pre-trained model:", MODEL_PATH)
    train_duration = 0

# === Model Analysis ===
print("\n=== Model Analysis ===")
print(f"Number of parameters: {model.count_params():,}")
print(f"Training duration: {train_duration:.2f}s" if train_duration > 0 else "Using pre-trained model")

# === Recursive Prediction Demo ===
def recursive_predict(model, initial_sequence, steps=100):
    """Recursive prediction function."""
    predictions = []
    current_sequence = initial_sequence.copy()
    
    for _ in range(steps):
        # Use the last 20 time steps' velocity for prediction
        velocity_sequence = current_sequence[-20:, 0]
        processed_seq = (velocity_sequence - data_mean[0]) / data_std[0]
        processed_seq = processed_seq.reshape(1, 20, 1)
        
        pred_outputs = model.predict(processed_seq, verbose=0)
        # De-normalize the outputs
        pred_velocity = pred_outputs[0][0][0] * data_std[0] + data_mean[0]
        pred_pressure = pred_outputs[1][0][0] * data_std[1] + data_mean[1]
        pred_diameter = pred_outputs[2][0][0] * data_std[2] + data_mean[2]
        
        new_point = np.array([pred_velocity, pred_pressure, pred_diameter])
        current_sequence = np.vstack([current_sequence, new_point])
        predictions.append(new_point)
    
    return np.array(predictions)

# Using recursive prediction to forecast future points
test_start_index = 800
initial_sequence = raw_data[test_start_index:test_start_index + 20]
predictions = recursive_predict(model, initial_sequence)

# === Results Evaluation and Visualization ===
pred_x = range(test_start_index + 20, test_start_index + 20 + len(predictions))
true_values = raw_data[test_start_index + 20 : test_start_index + 20 + len(predictions)]
eval_metrics = {
    'MSE': mean_squared_error(true_values, predictions),
    'MAE': mean_absolute_error(true_values, predictions),
    'R²': r2_score(true_values, predictions),
    'RMSE': np.sqrt(mean_squared_error(true_values, predictions)),
    'MAPE': np.mean(np.abs((true_values - predictions) / true_values)) * 100
}

plt.figure(figsize=(12, 6), dpi=100)
plt.title('LSTM Time Series Forecast Performance', fontsize=14)
plt.plot(raw_data, label='Original Data', color='#1f77b4', alpha=0.8)
plt.plot(pred_x, predictions, label='Predicted Results', linestyle='--', color='#ff7f0e')
metrics_text = "\n".join([f"{k}: {v:.4f}" if k != 'MAPE' else f"{k}: {v:.2f}%" 
                          for k, v in eval_metrics.items()])
plt.text(0.05, 0.7, metrics_text, transform=plt.gca().transAxes,
         bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10},
         fontsize=10, verticalalignment='top')
plt.axvline(test_start_index + 20, color='red', linestyle='--', label='Prediction Start')
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# === Additional Prediction Evaluation Function ===
def evaluate_on_new_data(model, data_mean, data_std, lookback=20):
    """Evaluate a new dataset using the pre-trained model."""
    # Generate new test data (using different parameters)
    new_data = generate_synthetic_data(timesteps=500)
    
    # Data preprocessing
    X_new, y_new = create_sequences(new_data, lookback)
    X_new_norm = (X_new - data_mean) / data_std
    X_new_norm = X_new_norm.reshape(-1, lookback, 1)
    
    start_time = time.time()
    predictions_norm = model.predict(X_new_norm, verbose=0)
    predictions = predictions_norm * data_std + data_mean
    infer_time = time.time() - start_time
    
    # Compute evaluation metrics
    eval_metrics = {
        'MSE': mean_squared_error(y_new, predictions),
        'MAE': mean_absolute_error(y_new, predictions),
        'R²': r2_score(y_new, predictions),
        'RMSE': np.sqrt(mean_squared_error(y_new, predictions)),
        'MAPE': np.mean(np.abs((y_new - predictions) / y_new)) * 100
    }
    
    # Compute hemodynamic indicators
    hemodynamics = evaluate_hemodynamics(predictions, y_new)
    
    # Visualize the comparison
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    plt.plot(new_data, label='New Dataset', alpha=0.7, color='#1f77b4')
    plt.plot(range(lookback, len(new_data)), predictions, 
             label='Model Prediction', linestyle='--', color='#ff7f0e')
    plt.title('Blood Flow Velocity: Predicted vs. True Values', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Velocity (m/s)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 1, 2)
    # Create a metrics table to display hemodynamic indicators
    cell_text = [
        [f"{hemodynamics['Systolic Error']:.3f} m/s", f"{hemodynamics['Diastolic Error']:.3f} m/s"],
        [f"{hemodynamics['Pulsatility Index']:.2f}", f"{hemodynamics['Mean Flow Velocity']:.3f} m/s"],
        [f"{hemodynamics['Spectral Entropy']:.2f} bits", ""]
    ]
    
    table = plt.table(cellText=cell_text,
                     colLabels=['Systolic/Diastolic Indicators', 'Flow Characteristics'],
                     rowLabels=['Peak Error', 'Pulsatility', 'Spectral Analysis'],
                     loc='center',
                     cellLoc='center',
                     colWidths=[0.3, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    plt.axis('off')
    plt.title('Hemodynamic Evaluation Metrics', fontsize=14, y=0.9)
    
    plt.tight_layout()
    plt.show()

def evaluate_hemodynamics(predictions, true_values):
    """Hemodynamic-specific evaluation metrics."""
    metrics = {
        'Systolic Error': np.max(true_values) - np.max(predictions),
        'Diastolic Error': np.min(true_values) - np.min(predictions),
        'Pulsatility Index': (np.max(predictions) - np.min(predictions)) / np.mean(predictions),
        'Mean Flow Velocity': np.mean(predictions),
        'Spectral Entropy': compute_spectral_entropy(predictions)
    }
    return metrics

def compute_spectral_entropy(signal, fs=100):
    """Compute the spectral entropy of the blood flow signal."""
    f, Pxx = welch(signal, fs=fs)
    psd_norm = Pxx / np.sum(Pxx)
    return -np.sum(psd_norm * np.log2(psd_norm))
