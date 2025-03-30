import os
import shutil

# Ensure that /opt/homebrew/bin appears in the PATH
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]
os.environ["GRAPHVIZ_DOT"] = "/opt/homebrew/bin/dot"

dot_path = shutil.which("dot")
print("dot found at:", dot_path)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import plot_model
import seaborn as sns
from tensorflow.keras.callbacks import TensorBoard
import datetime
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import applications
from keras_flops import get_flops
import warnings

# Suppress deprecation warnings from TensorFlow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# -----------------------------------------------------------------
# Generate synthetic time series data: sine wave with noise
# -----------------------------------------------------------------
def generate_time_series(num_points=1000):
    x = np.linspace(0, 20 * np.pi, num_points)
    y = np.sin(x) + 0.1 * np.random.randn(num_points)  # sine wave with added Gaussian noise
    return y.reshape(-1, 1)

# -----------------------------------------------------------------
# Create supervised learning dataset from time series data.
# This function uses a sliding window to form input sequences (X)
# and corresponding targets (Y).
# -----------------------------------------------------------------
def create_dataset(data, look_back=20):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : (i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# -----------------------------------------------------------------
# Data Preprocessing:
# - Generate synthetic data
# - Scale the data to the [0, 1] range using MinMaxScaler
# - Split data into training (80%) and testing (20%) sets
# -----------------------------------------------------------------
data = generate_time_series(1000)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Split the data into training and testing sets.
train_size = int(len(data_scaled) * 0.8)
train, test = data_scaled[:train_size], data_scaled[train_size:]

# Create supervised datasets (sliding window approach)
look_back = 24
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# Reshape the input to 3D tensor [samples, time steps, features] for LSTM compatibility.
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# -----------------------------------------------------------------
# Build the LSTM Model with a Convolution Layer and Residual Connections
# -----------------------------------------------------------------
inputs = keras.Input(shape=(look_back, 1))
x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

# First LSTM layer retains the sequence output (3D tensor).
lstm1 = LSTM(64, return_sequences=True)(x)

# Second LSTM layer is modified to output a sequence matching the first layer.
lstm2 = LSTM(64, return_sequences=True)(lstm1)  # Changing units from 32 to 64 for dimension consistency

# Add a residual connection to merge lstm1 and lstm2 (ensuring matching dimensions).
residual = keras.layers.Add()([lstm1, lstm2])

# Final LSTM layer outputs only the last time step (reducing dimensionality).
x = LSTM(16)(residual)
outputs = Dense(1)(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# -----------------------------------------------------------------
# Training Configuration:
# - Use EarlyStopping to monitor validation loss and prevent overfitting.
# - Use Adam optimizer with learning rate and gradient clipping.
# -----------------------------------------------------------------
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
opt = keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5)
model.compile(optimizer=opt, loss='mse')

# -----------------------------------------------------------------
# Add TensorBoard callback for training visualization
# -----------------------------------------------------------------
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

# -----------------------------------------------------------------
# Train the model using training data with a validation split.
# -----------------------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, tensorboard_cb],  # Include TensorBoard for monitoring training
    verbose=1,
)

# -----------------------------------------------------------------
# Define a function to visualize training history and prediction error distribution.
# -----------------------------------------------------------------
def visualize_training(history, y_true, y_pred):
    """Visualize training/validation loss curves and the error distribution of predictions."""
    plt.figure(figsize=(12, 6))
    
    # Plot training and validation losses over epochs.
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    # Plot the distribution of prediction errors (true - predicted).
    plt.subplot(1, 2, 2)
    errors = y_true.flatten() - y_pred.flatten()
    sns.histplot(errors, kde=True)
    plt.title("Prediction Error Distribution")
    plt.xlabel("Error")
    
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------
# Predict on the test set and inverse transform the outputs to their original scale.
# -----------------------------------------------------------------
test_predict = model.predict(X_test)
test_predict = scaler.inverse_transform(test_predict)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Visualize the training process and prediction errors.
visualize_training(history, y_test_actual, test_predict)

# -----------------------------------------------------------------
# Generate and save a diagram of the model's architecture.
# -----------------------------------------------------------------
plot_model(
    model,
    to_file="model_architecture.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",  # TB for a top-to-bottom view, LR for left-to-right
    dpi=96,
)

# -----------------------------------------------------------------
# Inter-Layer Activation Visualization:
# - Create a model that outputs activations at each layer.
# - Visualize the activations for a single test example.
# -----------------------------------------------------------------
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Get activations for the first test sample.
activations = activation_model.predict(X_test[:1])

# Plot the activation distributions for each layer.
plt.figure(figsize=(12, 8))
for i, (activation, layer) in enumerate(zip(activations, model.layers)):
    plt.subplot(len(activations), 1, i + 1)
    if len(activation.shape) == 3:
        # For 3D output tensors, use a matrix plot (transpose for clarity).
        plt.matshow(activation[0].T, cmap="viridis", fignum=False)
    else:
        # For 2D outputs, simply plot the activation values.
        plt.plot(activation[0])
    plt.title(f"{layer.name} ({layer.__class__.__name__})")
    plt.colorbar()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------
# Visualize final predictions vs. true values.
# -----------------------------------------------------------------
plt.figure(figsize=(12, 10))
plt.plot(y_test_actual, label="True Values")
plt.plot(test_predict, "--", label="Predictions")
plt.title("LSTM Time Series Prediction")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.legend()
plt.show()

# -----------------------------------------------------------------
# Compute evaluation metrics: Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
# -----------------------------------------------------------------
def calculate_metrics(actual, pred):
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mae = np.mean(np.abs(actual - pred))
    return rmse, mae

rmse, mae = calculate_metrics(y_test_actual, test_predict)
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

model.summary()

# Expected model summary output:
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 18, 64)            256       
                                                                 
 max_pooling1d (MaxPooling1D  (None, 9, 64)            0         
 )                                                               
                                                                 
 lstm (LSTM)                 (None, 9, 64)             33024     
                                                                 
 lstm_1 (LSTM)               (None, 64)                16448     
                                                                 
 dense (Dense)               (None, 1)                 65        
                                                                 
=================================================================
Total params: 71,761
Trainable params: 71,633
Non-trainable params: 128
"""

# -----------------------------------------------------------------
# Callback to log weight histograms to TensorBoard at the end of each epoch.
# Useful for monitoring the distribution of layer weights.
# -----------------------------------------------------------------
class WeightHistogram(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            # Check if the layer has kernel weights and log them.
            if "kernel" in layer.variables[0].name:
                tf.summary.histogram(f"{layer.name}/weights", layer.variables[0], step=epoch)

# -----------------------------------------------------------------
# Function to analyze gradients:
# - Computes the average gradient magnitude for each trainable weight.
# -----------------------------------------------------------------
def get_gradient_analysis(model, sample, y_true):
    # Create a MeanSquaredError loss function instance.
    loss_fn = tf.keras.losses.MeanSquaredError()
    with tf.GradientTape() as tape:
        output = model(sample)
        loss = loss_fn(y_true, output)
    grads = tape.gradient(loss, model.trainable_weights)
    return [K.mean(g).numpy() for g in grads]

# -----------------------------------------------------------------
# Visualize gradient distribution across layers.
# -----------------------------------------------------------------
gradients = get_gradient_analysis(model, X_test[:1], y_test[:1])  # Provide y_test sample for loss computation
plt.bar(range(len(gradients)), gradients)
plt.title("Gradient Magnitude per Layer")
plt.xlabel("Layer Index")
plt.ylabel("Average Gradient")
plt.show()

# -----------------------------------------------------------------
# Calculate and print total model parameters and estimated FLOPS.
# -----------------------------------------------------------------
total_params = model.count_params()
print(f"Total Parameters: {total_params:,}")

flops = get_flops(model, batch_size=1)
print(f"Estimated FLOPS: {flops/1e6:.1f} Million")