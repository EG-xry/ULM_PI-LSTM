"""
ULM PINN Tracking Framework
---------------------------
This script implements a physics-informed neural network (PINN) to post process
microbubble localization data (loaded from "mb_ulm_4con.csv") and reconstruct
a dense velocity and pressure field while simultaneously providing improved
tracking of microbubbles.

In this modified version:
    - No images or graphs are generated.
    - A new function 'extract_tracks' is added to group bubble positions into tracks.
    - The extracted tracks (only those with sufficient length) are written to a CSV file.
    - A new variable 'min_length' (default = 3) defines the minimal track length.
      
Author: Eric Gao (modified by Your Name)
Date: 2025-03-06
"""

import os           # Standard library for file and path operations
import argparse     # For command line argument parsing
import numpy as np  # For numerical operations
import pandas as pd # For data loading and for CSV file output

import torch        # PyTorch for deep learning and autograd
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms

# -----------------------------
# 1. Data Preprocessing Functions
# -----------------------------
def load_tracking_data(csv_file):
    """
    Load the microbubble tracking CSV file.
    
    Expected columns: Intensity, X, Y, ImageIndex. 
    - The coordinates are normalized using min-max normalization.
    - The ImageIndex is treated as a surrogate for time.
    - Finite differences are used to approximate velocities.
    
    Returns:
        X_norm: Normalized x-coordinates (np.array, shape (N,))
        Y_norm: Normalized y-coordinates (np.array, shape (N,))
        T_norm: Normalized time (np.array, shape (N,))
        measured_vel: Approximate velocities (np.array, shape (N,2))
    """
    df = pd.read_csv(csv_file)  # Read CSV into a DataFrame
    
    # Extract and cast spatial and temporal columns as float32
    X = df['X'].values.astype(np.float32)
    Y = df['Y'].values.astype(np.float32)
    T = df['ImageIndex'].values.astype(np.float32)  # Using frame index as a proxy for time

    # Sort the data by time to ensure proper finite difference computation
    sort_idx = np.argsort(T)
    X, Y, T = X[sort_idx], Y[sort_idx], T[sort_idx]

    # Approximate velocity using a central difference scheme (pad boundaries with zeros)
    u_meas = np.zeros_like(X)
    v_meas = np.zeros_like(Y)
    if len(T) > 2:
        # Central difference for internal indices (boundaries remain zero)
        u_meas[1:-1] = (X[2:] - X[:-2]) / (T[2:] - T[:-2] + 1e-6)
        v_meas[1:-1] = (Y[2:] - Y[:-2]) / (T[2:] - T[:-2] + 1e-6)
    measured_vel = np.stack([u_meas, v_meas], axis=-1)
    
    # Normalize the coordinates and time to the range [0,1]
    X_norm = (X - X.min()) / (X.max()-X.min() + 1e-6)
    Y_norm = (Y - Y.min()) / (Y.max()-Y.min() + 1e-6)
    T_norm = (T - T.min()) / (T.max()-T.min() + 1e-6)
    
    return X_norm, Y_norm, T_norm, measured_vel

def generate_collocation_points(n_points, domain):
    """
    Generate random collocation points within the given domain.
    
    Parameters:
        n_points: Number of collocation points (int)
        domain: Dictionary specifying the domain limits:
                {'x': (xmin, xmax), 'y': (ymin, ymax), 't': (tmin, tmax)}
    
    Returns:
        colloc_points: A numpy array of shape (n_points, 3), where each row is (x, y, t).
    """
    x = np.random.uniform(domain['x'][0], domain['x'][1], n_points)
    y = np.random.uniform(domain['y'][0], domain['y'][1], n_points)
    t = np.random.uniform(domain['t'][0], domain['t'][1], n_points)
    return np.stack([x, y, t], axis=-1)

# -----------------------------
# 2. PINN Network Definition
# -----------------------------
class PINN(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_layers=8, hidden_size=50):
        """
        A fully connected deep neural network using tanh activations.
        The PINN takes in features (x, y, t) and outputs (u, v, p), where:
            - u,v: Velocity components.
            - p: Pressure.
            
        Additionally, it defines a trainable log-viscosity parameter to be used
        in the physics-informed loss.
        """
        super(PINN, self).__init__()
        layers = []
        # Input layer: mapping input to first hidden layer
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.Tanh())
        # Adding hidden fully connected layers with tanh activation
        for _ in range(hidden_layers-1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        # Output layer: maps hidden layer to output (u, v, p)
        layers.append(nn.Linear(hidden_size, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Define a trainable parameter for log-viscosity ensuring positivity via exponentiation
        self.log_mu = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        
    def forward(self, x):
        """
        Forward pass: Process input x to produce outputs.
        Input:
            x: Tensor of shape (N, 3) for (x, y, t)
        Returns:
            u: Velocity component along x (Tensor of shape (N, 1))
            v: Velocity component along y (Tensor of shape (N, 1))
            p: Pressure (Tensor of shape (N, 1))
        """
        out = self.network(x)
        # Split the network output into u, v, and p components
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        return u, v, p
    
    def get_mu(self):
        """
        Return the dynamic viscosity (mu) by exponentiating the log_mu parameter.
        """
        return torch.exp(self.log_mu)

# -----------------------------
# 3. Loss Functions
# -----------------------------
def data_loss(model, x_data, measured_vel):
    """
    Compute the mean squared error between the measured and predicted velocities.
    
    Parameters:
        model: The PINN model.
        x_data: Tensor of shape (N,3) containing measured (x,y,t).
        measured_vel: Tensor of shape (N,2) containing measured velocities (u,v).
    
    Returns:
        loss: A scalar tensor representing the data loss.
    """
    u_pred, v_pred, _ = model(x_data)
    pred_vel = torch.cat((u_pred, v_pred), dim=1)
    loss = torch.mean((pred_vel - measured_vel) ** 2)
    return loss

def physics_loss(model, x_colloc):
    """
    Compute the physics-informed loss by enforcing the PDE constraints via
    automatic differentiation.
    
    Enforces the conservation of momentum and mass (i.e., incompressibility):
        - R_u = -p_x + mu*(u_xx + u_yy)
        - R_v = -p_y + mu*(v_xx + v_yy)
        - R_div = u_x + v_y
        
    Parameters:
        model: The PINN model.
        x_colloc: Collocation points tensor of shape (N,3) (requires gradient).
    
    Returns:
        loss_phys: A scalar tensor representing the physics loss.
    """
    x_colloc.requires_grad_(True)
    u, v, p = model(x_colloc)
    mu = model.get_mu()
    
    grads = {}
    for var, name in zip([u, v, p], ['u', 'v', 'p']):
        grad = torch.autograd.grad(var, x_colloc, grad_outputs=torch.ones_like(var),
                                   retain_graph=True, create_graph=True)[0]
        grads[name] = grad  # Each grad is of shape (N, 3) for (d/dx, d/dy, d/dt)
    
    u_x = grads['u'][:, 0:1]
    u_y = grads['u'][:, 1:2]
    v_x = grads['v'][:, 0:1]
    v_y = grads['v'][:, 1:2]
    p_x = grads['p'][:, 0:1]
    p_y = grads['p'][:, 1:2]
    
    # Compute second order derivatives (Laplacian components)
    u_xx = torch.autograd.grad(u_x, x_colloc, grad_outputs=torch.ones_like(u_x),
                               retain_graph=True, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, x_colloc, grad_outputs=torch.ones_like(u_y),
                               retain_graph=True, create_graph=True)[0][:, 1:2]
    v_xx = torch.autograd.grad(v_x, x_colloc, grad_outputs=torch.ones_like(v_x),
                               retain_graph=True, create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y, x_colloc, grad_outputs=torch.ones_like(v_y),
                               retain_graph=True, create_graph=True)[0][:, 1:2]
    
    R_u = -p_x + mu * (u_xx + u_yy)
    R_v = -p_y + mu * (v_xx + v_yy)
    R_div = u_x + v_y  # Enforcing divergence free condition
    
    loss_phys = torch.mean(R_u ** 2) + torch.mean(R_v ** 2) + torch.mean(R_div ** 2)
    return loss_phys

# -----------------------------
# 4. Training Function
# -----------------------------
def train_pinn(model, optimizer, x_data, measured_vel, domain,
               n_colloc=1000, beta=1.0, epochs=5000, print_every=500):
    """
    Train the PINN model by minimizing a combined loss:
        - Data loss: mismatch between predicted and measured velocities.
        - Physics loss: residuals of the governing PDEs.
    
    Parameters:
        model: The PINN model.
        optimizer: The optimizer for updating model parameters.
        x_data: Tensor of shape (N,3) of measured coordinates.
        measured_vel: Tensor of shape (N,2) of measured velocities.
        domain: Domain limits for collocation points.
        n_colloc: Number of collocation points sampled per epoch.
        beta: Scaling factor for the physics loss.
        epochs: Total number of training epochs.
        print_every: Frequency of printing training losses.
    
    Returns:
        loss_history: List of total loss values per epoch.
    """
    model.train()
    loss_history = []
    for ep in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Compute data loss on measured positions
        loss_d = data_loss(model, x_data, measured_vel)
        
        # Generate collocation points and compute physics loss
        colloc_np = generate_collocation_points(n_colloc, domain)
        x_colloc = torch.tensor(colloc_np, dtype=torch.float32, device=x_data.device)
        loss_p = physics_loss(model, x_colloc)
        
        loss_total = loss_d + beta * loss_p
        loss_total.backward()
        optimizer.step()
        
        loss_history.append(loss_total.item())
        if ep % print_every == 0:
            print(f"Epoch {ep:05d}: Total Loss = {loss_total.item():.4e}, Data Loss = {loss_d.item():.4e}, Physics Loss = {loss_p.item():.4e}, mu = {model.get_mu().item():.4e}")
    return loss_history

# -----------------------------
# 5. Track Extraction Function
# -----------------------------
def extract_tracks(positions, time_round=2, distance_threshold=0.05):
    """
    Extract bubble tracks from position measurements.
    
    The function groups detections by frame based on the rounded time value
    and links detections in consecutive frames if they lie within a specified spatial
    distance threshold.
    
    Parameters:
        positions: numpy array of shape (N,3), where each row is (x, y, t).
                   Time is expected to be normalized.
        time_round: Number of decimals to round the time, used for grouping detections per frame.
        distance_threshold: Maximum allowed spatial distance to link detections between frames.
    
    Returns:
        tracks: A list of tracks. Each track is a list of tuples (x, y, t), representing
                the coordinates of a bubble over time.
    """
    # Group detections by frame (using rounded time)
    groups = {}
    for pos in positions:
        frame = round(pos[2], time_round)
        if frame not in groups:
            groups[frame] = []
        groups[frame].append((pos[0], pos[1], pos[2]))
    
    # Sort the frames chronologically
    frames = sorted(groups.keys())
    
    # Initialize tracks using detections from the first frame
    tracks = []
    if frames:
        for coord in groups[frames[0]]:
            tracks.append([coord])
    
    # Process subsequent frames to link detections to existing tracks
    for frame in frames[1:]:
        detections = groups[frame]
        assigned = [False] * len(detections)
        
        # Try to associate each detection with an existing track based on closeness.
        for track in tracks:
            last_coord = track[-1]
            best_det_idx = None
            best_distance = float('inf')
            for i, det in enumerate(detections):
                if not assigned[i] and det[2] > last_coord[2]:
                    # Compute Euclidean distance between last point in track and detection
                    d = np.hypot(det[0] - last_coord[0], det[1] - last_coord[1])
                    if d < distance_threshold and d < best_distance:
                        best_distance = d
                        best_det_idx = i
            if best_det_idx is not None:
                track.append(detections[best_det_idx])
                assigned[best_det_idx] = True
        
        # Create new tracks for detections that were not assigned
        for i, det in enumerate(detections):
            if not assigned[i]:
                tracks.append([det])
    
    return tracks

# -----------------------------
# 6. Function to Write Tracks to CSV File
# -----------------------------
def write_tracks_csv(tracks, filename, min_length):
    """
    Write the extracted tracks to a CSV file.
    
    Only tracks that have a length equal or greater than 'min_length' are saved.
    Each output row in the CSV contains:
        - track_id: Identifier for the track.
        - point_index: The index of the point within the track.
        - x: x-coordinate.
        - y: y-coordinate.
        - t: time.
    
    Parameters:
        tracks: List of tracks (each track is a list of tuples (x, y, t)).
        filename: Path to the CSV file to be written.
        min_length: Minimal allowed length for a track to be saved.
    """
    rows = []
    for track_id, track in enumerate(tracks, start=1):
        if len(track) < min_length:
            continue  # Skip tracks that are too short.
        for index, (x, y, t) in enumerate(track):
            rows.append({
                "track_id": track_id,
                "point_index": index,
                "x": x,
                "y": y,
                "t": t
            })
    df_tracks = pd.DataFrame(rows)
    df_tracks.to_csv(filename, index=False)
    print(f"Tracks saved to {filename}")

# -----------------------------
# 7. Main Function and Argument Parsing
# -----------------------------
def main(args):
    # Set up computation device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # ----------------- Data Loading -----------------
    # Load tracking data from the CSV file (absolute path is provided)
    csv_file = args.csv_file
    X, Y, T, measured_vel_np = load_tracking_data(csv_file)
    print(f"Loaded {len(X)} data points.")
    
    # Create data tensors for input into the network
    x_data_np = np.stack([X, Y, T], axis=-1)
    x_data = torch.tensor(x_data_np, dtype=torch.float32, device=device)
    measured_vel = torch.tensor(measured_vel_np, dtype=torch.float32, device=device)
    
    # Define the normalized domain for collocation points
    domain = {
        'x': (0.0, 1.0),
        'y': (0.0, 1.0),
        't': (0.0, 1.0)
    }
    
    # ----------------- Model Initialization and Training -----------------
    model = PINN(input_dim=3, output_dim=3, hidden_layers=args.hidden_layers, hidden_size=args.hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("Training PINN...")
    loss_history = train_pinn(model, optimizer, x_data, measured_vel, domain,
                              n_colloc=args.n_colloc, beta=args.beta, epochs=args.epochs, print_every=args.print_every)
    print("Training complete!")
    print(f"Trainable viscosity mu: {model.get_mu().item():.4e}")
    
    # ----------------- Track Extraction -----------------
    # Extract tracks from the measured bubble positions
    tracks = extract_tracks(x_data_np, time_round=2, distance_threshold=0.05)
    
    # Write the tracks to a CSV file, filtering out those with length below the specified minimum.
    write_tracks_csv(tracks, args.tracks_csv, args.min_length)

# -----------------------------
# Argument Parsing and Script Execution
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ULM Pipeline PINN Framework")
    # Use the absolute path to the CSV file as the default.
    parser.add_argument("--csv_file", type=str, default="/Users/eric/Desktop/ULM_PI-LSTM/TimeSeries-Forecast/PALA_scripts/lstm/mb_ulm_4con.csv", 
                        help="Absolute path to microbubble tracking CSV file")
    parser.add_argument("--hidden_layers", type=int, default=8, help="Number of hidden layers in the network")
    parser.add_argument("--hidden_size", type=int, default=50, help="Number of neurons per hidden layer")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--n_colloc", type=int, default=1000, help="Number of collocation points")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for the physics loss")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument("--print_every", type=int, default=500, help="Frequency of printing training info")
    # New argument for minimal track length (ULM.min_length) with a default value of 3
    parser.add_argument("--min_length", type=int, default=3, help="Minimal length of tracks to be saved")
    # CSV filename for output tracks
    parser.add_argument("--tracks_csv", type=str, default="tracks.csv", help="Filename for the output tracks CSV file")
    args = parser.parse_args()
    
    main(args)