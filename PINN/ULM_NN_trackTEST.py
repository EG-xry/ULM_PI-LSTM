import os           # Standard library for file and path operations
import argparse     # For command line argument parsing
import numpy as np  # For numerical operations
import pandas as pd # For data loading and for CSV file outpu
from scipy.interpolate import interp1d  # For track interpolation
from scipy.signal import savgol_filter  # For smoothing (similar to MATLAB's smooth)

import torch        # PyTorch for deep learning and autograd
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms

from tqdm import tqdm  # Added progress bar support

# -----------------------------
# New Activation Modules for SIREN
# -----------------------------
class SineFirst(nn.Module):
    """
    The first layer in a SIREN network.
    Applies a linear transformation and then a sine activation with scaling factor omega_0.
    """
    def __init__(self, in_features, out_features, omega_0=30):
        super(SineFirst, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
        
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class Sine(nn.Module):
    """
    Subsequent layers in a SIREN network.
    Applies a sine activation (here with a multiplier of 1).
    """
    def __init__(self, omega_0=1.0):
        super(Sine, self).__init__()
        self.omega_0 = omega_0
        
    def forward(self, x):
        return torch.sin(self.omega_0 * x)

class FourierFeatureMapping(nn.Module):
    """
    Implements random Fourier feature mapping for better fitting of high-frequency functions.
    This creates a fixed random encoding of the input to help the network learn spatial patterns.
    """
    def __init__(self, input_dim=3, mapping_size=16, scale=10.0):
        super(FourierFeatureMapping, self).__init__()
        # Initialize random weights for Fourier features
        self.register_buffer('B', torch.randn((input_dim, mapping_size)) * scale)
        
    def forward(self, x):
        # Project input onto random Fourier basis and return sine/cosine features
        x_proj = 2 * np.pi * torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# -----------------------------
# 1. Data Preprocessing Functions
# -----------------------------
def load_tracking_data(csv_file, velocity_clip=0.1):
    """
    Load the microbubble tracking CSV file.
    
    Expected columns: Intensity, X, Z, ImageIndex. 
    - The coordinates (X, Z) are normalized to the range [0,1].
    - The ImageIndex is used as the frame number and converted into a positive integer (starting at 1).
    - Measured velocities are calculated using central differences, and then
      robustly scaled and clipped to ensure reasonable magnitudes.
      
    Returns:
        X_norm: Normalized x-coordinates (np.array, shape (N,))
        Z_norm: Normalized z-coordinates (np.array, shape (N,))
        T_norm: Normalized time (np.array, shape (N,)) in the range [0,1]
        measured_vel: Appropriately scaled velocities (np.array, shape (N,2))
        orig_bounds: Dictionary containing original coordinate bounds for unnormalization
    """
    df = pd.read_csv(csv_file)  # Read CSV into a DataFrame
    
    # Extract original spatial and temporal data
    X = df['X'].values.astype(np.float32)
    Z = df['Z'].values.astype(np.float32)  # Corrected: using 'Z' column from CSV
    T = df['ImageIndex'].values   # Keep original time values from the CSV
    
    # Sort the data by time (frame) to ensure proper finite difference computation
    sort_idx = np.argsort(T)
    X = X[sort_idx]
    Z = Z[sort_idx]
    T = T[sort_idx]
    
    # Normalize the spatial coordinates to the range [0,1]
    X_min, X_max = X.min(), X.max()
    Z_min, Z_max = Z.min(), Z.max()
    X_norm = (X - X_min) / (X_max - X_min + 1e-6)
    Z_norm = (Z - Z_min) / (Z_max - Z_min + 1e-6)
    
    # Store original bounds for unnormalization later
    orig_bounds = {
        'X_min': X_min, 'X_max': X_max,
        'Z_min': Z_min, 'Z_max': Z_max
    }
    
    # Convert ImageIndex to positive integers starting at 1
    T_int = T.astype(np.int32)
    T_frame = T_int - T_int.min() + 1  # Ensures frames start at 1
    
    # Normalize the time coordinate to [0,1]
    T_norm = (T_frame - T_frame.min()) / (T_frame.max() - T_frame.min() + 1e-6)
    
    # Update orig_bounds with time limits for later unnormalization
    orig_bounds['T_min'] = T_frame.min()
    orig_bounds['T_max'] = T_frame.max()
    
    # Compute raw velocities using central differences
    u_raw = np.zeros_like(X_norm)
    v_raw = np.zeros_like(Z_norm)
    if len(T_frame) > 2:
        # To avoid integer division, cast time differences to float
        dt = (T_frame[2:] - T_frame[:-2]).astype(np.float32)
        u_raw[1:-1] = (X_norm[2:] - X_norm[:-2]) / (dt + 1e-6)
        v_raw[1:-1] = (Z_norm[2:] - Z_norm[:-2]) / (dt + 1e-6)
    
    # Robust scaling of velocities to handle outliers
    u_raw = np.nan_to_num(u_raw, nan=0.0, posinf=velocity_clip, neginf=-velocity_clip)
    v_raw = np.nan_to_num(v_raw, nan=0.0, posinf=velocity_clip, neginf=-velocity_clip)
    u_raw = np.clip(u_raw, -velocity_clip, velocity_clip)
    v_raw = np.clip(v_raw, -velocity_clip, velocity_clip)
    u_scaled = u_raw / (np.max(np.abs(u_raw)) + 1e-6)
    v_scaled = v_raw / (np.max(np.abs(v_raw)) + 1e-6)
    
    # Stack the velocity components
    measured_vel = np.stack([u_scaled, v_scaled], axis=-1)
    
    print(f"Velocity stats - Min: {measured_vel.min():.4f}, Max: {measured_vel.max():.4f}, Mean: {measured_vel.mean():.4f}")
    
    return X_norm, Z_norm, T_norm, measured_vel, orig_bounds

def generate_collocation_points(n_points, domain):
    """
    Generate random collocation points within the given domain.
    
    Parameters:
        n_points: Number of collocation points (int)
        domain: Dictionary specifying the domain limits:
                {'x': (xmin, xmax), 'z': (zmin, zmax), 't': (tmin, tmax)}
    
    Returns:
        colloc_points: A numpy array of shape (n_points, 3), where each row is (x, z, t).
    """
    x = np.random.uniform(domain['x'][0], domain['x'][1], n_points)
    z = np.random.uniform(domain['z'][0], domain['z'][1], n_points)
    t = np.random.uniform(domain['t'][0], domain['t'][1], n_points)
    return np.stack([x, z, t], axis=-1)

# -----------------------------
# 2. PINN Network Definition
# -----------------------------
class PINN(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_layers=10, hidden_size=128,
                 activation='sine', omega_0=30,
                 use_fourier_features=False, mapping_size=16, fourier_scale=10.0):
        """
        A fully connected deep neural network that supports either sine or tanh activations.
        Optionally, it embeds inputs using Fourier feature mapping.
        
        The PINN takes in features (x, z, t) and outputs (u, v, p), where:
            - u, v: Velocity components.
            - p: Pressure.
        
        It also defines a trainable log-viscosity parameter.
        
        Parameters:
            activation: 'sine' or 'tanh'.
            omega_0: Scaling factor for the first layer (for sine activations).
            use_fourier_features: If True, the input is first passed through a Fourier feature mapping.
            mapping_size: The number of Fourier features.
            fourier_scale: Scaling factor for the Fourier features.
            hidden_layers: Number of hidden layers in the network.
            hidden_size: Number of neurons per hidden layer.
        """
        super(PINN, self).__init__()
        self.activation = activation
        self.omega_0 = omega_0
        self.use_fourier_features = use_fourier_features

        if self.use_fourier_features:
            # Apply Fourier feature mapping to the input
            self.fourier = FourierFeatureMapping(input_dim=input_dim, mapping_size=mapping_size, scale=fourier_scale)
            effective_input_dim = mapping_size * 2
        else:
            effective_input_dim = input_dim

        layers = []
        if activation == 'sine':
            # SIREN-style network
            layers.append(SineFirst(effective_input_dim, hidden_size, omega_0))
            for _ in range(hidden_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(Sine(omega_0=1.0))
            layers.append(nn.Linear(hidden_size, output_dim))
        else:
            # Default architecture with tanh activations.
            layers.append(nn.Linear(effective_input_dim, hidden_size))
            layers.append(nn.Tanh())
            for _ in range(hidden_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_size, output_dim))
            
        self.network = nn.Sequential(*layers)
        
        # Trainable log-viscosity parameter.
        self.log_mu = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        # Trainable parameters for adaptive loss weighting (uncertainty weighting from Kendall et al.)
        self.log_sigma_data = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.log_sigma_phys = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        
        # Initialize network weights with a scheme suited to the activation type.
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights depending on the chosen activation function."""
        if self.activation == 'sine':
            first_linear_found = False
            for m in self.network.modules():
                if isinstance(m, nn.Linear):
                    if not first_linear_found:
                        # First linear layer in SIREN: narrow initialization.
                        nn.init.uniform_(m.weight, -1.0 / m.in_features, 1.0 / m.in_features)
                        first_linear_found = True
                    else:
                        # Subsequent layers: uniform initialization with bounds scaled by omega_0.
                        bound = np.sqrt(6 / m.in_features) / self.omega_0
                        nn.init.uniform_(m.weight, -bound, bound)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            # Xavier initialization for tanh-based networks.
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
    def forward(self, x):
        """
        Forward pass: if Fourier features are enabled, map the inputs, then compute (u, v, p).
        
        Input:
            x: Tensor of shape (N, 3) corresponding to (x, z, t).
        Returns:
            u: Velocity component along x.
            v: Velocity component along z.
            p: Pressure.
        """
        if self.use_fourier_features:
            x = self.fourier(x)
        out = self.network(x)
        # Split output into (u, v, p).
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        return u, v, p
    
    def get_mu(self):
        """
        Compute the dynamic viscosity (mu) from the trainable parameter.
        """
        return torch.exp(self.log_mu)

# -----------------------------
# 3. Loss Functions
# -----------------------------
def force(x, z):
    """
    Define the forcing function f(x,z).
    x, z: Tensors of shape (N,1) corresponding to the spatial coordinates.
    
    For this example, we assume zero forcing.
    Modify this function to impose a different force field if needed.
    """
    f_x = torch.zeros_like(x)
    f_z = torch.zeros_like(z)
    return f_x, f_z

def physics_loss(model, x_colloc):
    """
    Compute the physics loss enforcing the simplified Navier–Stokes (Stokes flow)
    equations:
        -∇p + μ∇²u + f = 0,    ∇⋅u = 0.
        
    Inputs:
      - model: The PINN model that predicts u, v, and p.
      - x_colloc: Tensor of shape (N_eq, 3) representing collocation points (x, z, t).
      
    Returns:
      - loss_phys: A scalar tensor representing the physics loss.
    """
    # Enable gradients for collocation points:
    x_colloc.requires_grad_(True)
    
    # Predict u, v, p at the collocation points.
    u, v, p = model(x_colloc)
    mu = model.get_mu()  # μ = exp(log_mu) is computed here.
    
    # Compute first derivatives:
    grads_u = torch.autograd.grad(u, x_colloc,
                                  grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
    grads_v = torch.autograd.grad(v, x_colloc,
                                  grad_outputs=torch.ones_like(v),
                                  retain_graph=True, create_graph=True)[0]
    grads_p = torch.autograd.grad(p, x_colloc,
                                  grad_outputs=torch.ones_like(p),
                                  retain_graph=True, create_graph=True)[0]
    
    # Extract derivatives with respect to x and z.
    u_x = grads_u[:, 0:1]
    u_z = grads_u[:, 1:2]
    v_x = grads_v[:, 0:1]
    v_z = grads_v[:, 1:2]
    p_x = grads_p[:, 0:1]
    p_z = grads_p[:, 1:2]

    # Compute second derivatives: needed for the Laplacian of u and v.
    u_xx = torch.autograd.grad(u_x, x_colloc,
                               grad_outputs=torch.ones_like(u_x),
                               retain_graph=True, create_graph=True)[0][:, 0:1]
    u_zz = torch.autograd.grad(u_z, x_colloc,
                               grad_outputs=torch.ones_like(u_z),
                               retain_graph=True, create_graph=True)[0][:, 1:2]
    v_xx = torch.autograd.grad(v_x, x_colloc,
                               grad_outputs=torch.ones_like(v_x),
                               retain_graph=True, create_graph=True)[0][:, 0:1]
    v_zz = torch.autograd.grad(v_z, x_colloc,
                               grad_outputs=torch.ones_like(v_z),
                               retain_graph=True, create_graph=True)[0][:, 1:2]
    
    # Laplacians of u and v:
    laplacian_u = u_xx + u_zz
    laplacian_v = v_xx + v_zz

    # Evaluate the force f(x,z) at the collocation points. Assume forcing depends only on spatial coordinates:
    f_x, f_z = force(x_colloc[:, 0:1], x_colloc[:, 1:2])
    
    # Compute momentum residual components:
    e_mom_x = -p_x + mu * laplacian_u + f_x
    e_mom_z = -p_z + mu * laplacian_v + f_z
    
    # Norm squared of the momentum residual (for each collocation point):
    e_mom_norm = e_mom_x**2 + e_mom_z**2
    
    # Compute the divergence (continuity) residual:
    e_div = u_x + v_z
    
    # Compute losses:
    loss_mom = torch.mean(e_mom_norm)
    loss_div = torch.mean(e_div**2)
    
    # Total physics loss:
    loss_phys = loss_mom + loss_div
    return loss_phys

def data_loss(model, x_data, measured_vel):
    """
    Compute the data loss:
      L_data = (1/N_data) ∑ || d𝑥/dt - u(x,z,t) ||²
      
    Here, measured_vel represents d𝑥/dt computed from the tracking data.
    """
    u_pred, v_pred, _ = model(x_data)
    pred_vel = torch.cat((u_pred, v_pred), dim=1)
    loss = torch.mean((pred_vel - measured_vel)**2)
    return loss

def total_loss(model, x_data, measured_vel, x_colloc, beta, adaptive_loss):
    """
    Compute the total loss.
    
    In the adaptive loss formulation (from Kendall et al.), one typically has:
      L_total = L_data/(2σ_data²) + (β L_phys)/(2σ_phys²) + log(σ_data) + log(σ_phys)
      
    To force the data loss to be minimized more aggressively, we remove the adaptive weighting
    on the data loss term while keeping the adaptive weighting for the physics loss.
    """
    loss_d = data_loss(model, x_data, measured_vel)
    loss_p = physics_loss(model, x_colloc)
    if adaptive_loss:
        # Note: data loss is not weighted adaptively (i.e. its weight is fixed to 1)
        loss_total = loss_d + 0.5 * (torch.exp(-2 * model.log_sigma_phys) * (beta * loss_p)) + model.log_sigma_phys
    else:
        loss_total = loss_d + beta * loss_p
    return loss_total

# -----------------------------
# 4. Training Function
# -----------------------------
def train_pinn(model, optimizer, 
               x_data, measured_vel, domain,
               n_colloc=1000,
               beta=1.0, epochs=3000, print_every=500,
               scheduler=None, adaptive_loss=False):
    """
    Training procedure using ADAM optimizer
      
    Parameters:
      model: The PINN model.
      optimizer: ADAM optimizer instance.
      x_data: Tensor of shape (N,3) of measured coordinates.
      measured_vel: Tensor of shape (N,2) of measured velocities.
      domain: Domain limits for collocation points.
      n_colloc: Number of collocation points during training.
      beta: Weight for the physics loss.
      epochs: Number of training epochs.
      print_every: Frequency of printing training information.
      scheduler: Optional learning rate scheduler.
      adaptive_loss: Boolean flag to use adaptive loss weighting.
      
    Returns:
      loss_history: List of loss values during training.
    """
    model.train()
    loss_history = []
    
    print("Training with ADAM (data loss only)")
    for ep in range(1, epochs+1):
        optimizer.zero_grad()
        
        # Compute data loss only (ignoring physics loss)
        loss_d = data_loss(model, x_data, measured_vel)
        loss_total = loss_d
        
        loss_total.backward()
        optimizer.step()
        
        # Add scheduler step if provided
        if scheduler is not None:
            scheduler.step(loss_total)
        
        loss_history.append(loss_total.item())
        if ep % print_every == 0:
            print(f"Epoch {ep:05d}: Data Loss = {loss_d.item():.4e}")
    
    return loss_history

# -----------------------------
# 5. Track Extraction Function
# -----------------------------
def extract_tracks(positions, time_round=2, distance_threshold=0.05):
    """
    Extract bubble tracks from position measurements.
    
    The function groups detections by frame (using rounded time) and links
    detections in consecutive frames if they lie within a specified spatial
    distance threshold.
    
    Parameters:
        positions: numpy array of shape (N,3), where each row is (x, z, t).
                   Time is expected to be standardized.
        time_round: Number of decimals to round the time.
        distance_threshold: Max allowed spatial distance for linking detections.
    
    Returns:
        tracks: A list of tracks (each track is a list of tuples (x, z, t)).
    """
    groups = {}
    for pos in positions:
        frame = round(pos[2], time_round)
        if frame not in groups:
            groups[frame] = []
        groups[frame].append((pos[0], pos[1], pos[2]))
    
    frames = sorted(groups.keys())
    tracks = []
    if frames:
        for coord in groups[frames[0]]:
            tracks.append([coord])
    
    for frame in frames[1:]:
        detections = groups[frame]
        assigned = [False] * len(detections)
        
        for track in tracks:
            last_coord = track[-1]
            best_det_idx = None
            best_distance = float('inf')
            for i, det in enumerate(detections):
                if not assigned[i] and det[2] > last_coord[2]:
                    d = np.hypot(det[0] - last_coord[0], det[1] - last_coord[1])
                    if d < distance_threshold and d < best_distance:
                        best_distance = d
                        best_det_idx = i
            if best_det_idx is not None:
                track.append(detections[best_det_idx])
                assigned[best_det_idx] = True
        
        for i, det in enumerate(detections):
            if not assigned[i]:
                tracks.append([det])
    
    return tracks

# -----------------------------
# 6. Track Interpolation and CSV Writing
# -----------------------------
def interpolate_tracks(tracks, interp_factor=5, smooth_factor=20):
    """
    Interpolate tracks with smoothing similar to the MATLAB implementation.
    
    Parameters:
        tracks: List of tracks (each track is a list of tuples (x, z, t))
        interp_factor: (Unused in the dynamic version)
        smooth_factor: Window size for Savitzky-Golay filter (equivalent to MATLAB's smooth)
    
    Returns:
        interp_tracks: List of interpolated tracks (each track is a list of tuples (x, z, t))
    """
    interp_tracks = []
    
    # Compute the dynamic interpolation factor using ULM parameters:
    ULM_max_linking_distance = 10   # as provided
    res = 1.0                       # default resolution
    dynamic_interp = 1 / (ULM_max_linking_distance * res) * 0.8  # e.g. 1/(10*1)*0.8 = 0.08

    # Use 1/dynamic_interp more points than the original:
    for track in tqdm(tracks, desc="Interpolating tracks"):
        if len(track) < 3:  # Need at least 3 points for meaningful interpolation
            continue
            
        # Extract coordinates
        track_array = np.array(track)
        x = track_array[:, 0]
        z = track_array[:, 1]
        t = track_array[:, 2]
        
        # Original indices
        indices = np.arange(len(x))
        
        # New number of indices for interpolation using the dynamic factor
        new_num_points = int(len(x) * (1 / dynamic_interp))
        new_indices = np.linspace(0, len(x)-1, new_num_points)
        
        # Smoothing - similar to MATLAB's smooth function
        # For short tracks, reduce smooth_factor to avoid errors
        actual_smooth = min(smooth_factor, len(x)-1 if len(x) % 2 == 0 else len(x))
        if actual_smooth > 2:  # Need at least 3 points for smoothing
            # Ensure actual_smooth is odd (required for savgol_filter)
            if actual_smooth % 2 == 0:
                actual_smooth -= 1
            
            if actual_smooth >= 3:  # Minimum window size for savgol_filter
                x_smooth = savgol_filter(x, actual_smooth, 2)
                z_smooth = savgol_filter(z, actual_smooth, 2)
            else:
                x_smooth, z_smooth = x, z
        else:
            x_smooth, z_smooth = x, z
        
        # Interpolation
        interp_x = interp1d(indices, x_smooth, kind='linear', bounds_error=False, fill_value="extrapolate")(new_indices)
        interp_z = interp1d(indices, z_smooth, kind='linear', bounds_error=False, fill_value="extrapolate")(new_indices)
        interp_t = interp1d(indices, t, kind='linear', bounds_error=False, fill_value="extrapolate")(new_indices)
        
        # Create new track with interpolated points
        interp_track = [(x_i, z_i, t_i) for x_i, z_i, t_i in zip(interp_x, interp_z, interp_t)]
        interp_tracks.append(interp_track)
    
    return interp_tracks

def unnormalize_tracks(tracks, orig_bounds):
    """
    Unnormalize track coordinates using the original bounds.
    
    Parameters:
        tracks: List of tracks (each track is a list of tuples (x, z, t))
        orig_bounds: Dictionary with original coordinate bounds
    
    Returns:
        unnorm_tracks: List of tracks with unnormalized coordinates
    """
    X_min, X_max = orig_bounds['X_min'], orig_bounds['X_max']
    Z_min, Z_max = orig_bounds['Z_min'], orig_bounds['Z_max']
    T_min, T_max = orig_bounds['T_min'], orig_bounds['T_max']
    
    unnorm_tracks = []
    for track in tqdm(tracks, desc="Unnormalizing tracks"):
        unnorm_track = []
        for x, z, t in track:
            # Unnormalize x and z coordinates
            x_unnorm = x * (X_max - X_min) + X_min
            z_unnorm = z * (Z_max - Z_min) + Z_min
            # Unnormalize time (t is normalized to [0,1])
            t_unnorm = t * (T_max - T_min) + T_min
            unnorm_track.append((x_unnorm, z_unnorm, t_unnorm))
        unnorm_tracks.append(unnorm_track)
    
    return unnorm_tracks

def write_tracks_csv(tracks, filename, min_length, include_velocity=False):
    """
    Write the extracted tracks to a CSV file.
    
    Only tracks with length >= min_length are written.
    
    Parameters:
        tracks: List of tracks (each track is a list of tuples (x, z, t)).
        filename: Path to the output CSV file.
        min_length: Minimum track length to be saved.
        include_velocity: Whether to calculate and include velocity in the output.
    """
    rows = []
    for track_id, track in tqdm(enumerate(tracks, start=1), total=len(tracks), desc="Writing tracks to CSV"):
        if len(track) < min_length:
            continue
        
        # If velocity is requested, calculate it
        if include_velocity and len(track) > 1:
            # Calculate velocities (forward difference)
            vx = np.zeros(len(track))
            vz = np.zeros(len(track))
            
            for i in range(len(track)-1):
                dt = track[i+1][2] - track[i][2]
                if dt > 0:
                    vx[i] = (track[i+1][0] - track[i][0]) / dt
                    vz[i] = (track[i+1][1] - track[i][1]) / dt
            
            # Last point velocity (use the previous one)
            vx[-1] = vx[-2] if len(track) > 1 else 0
            vz[-1] = vz[-2] if len(track) > 1 else 0
            
            for index, ((x, z, t), vx_val, vz_val) in enumerate(zip(track, vx, vz)):
                rows.append({
                    "track_id": track_id,
                    "point_index": index + 1,
                    "x": x,
                    "z": z,
                    "t": t,
                    "vx": vx_val,
                    "vz": vz_val
                })
        else:
            for index, (x, z, t) in enumerate(track):
                rows.append({
                    "track_id": track_id,
                    "point_index": index + 1,
                    "x": x,
                    "z": z,
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
    csv_file = args.csv_file
    X, Z, T_norm, measured_vel_np, orig_bounds = load_tracking_data(csv_file, velocity_clip=args.velocity_clip)
    print(f"Loaded {len(X)} data points.")
    
    # Create data tensors for input into the network (using normalized time)
    x_data_np = np.stack([X, Z, T_norm], axis=-1)
    x_data = torch.tensor(x_data_np, dtype=torch.float32, device=device)
    measured_vel = torch.tensor(measured_vel_np, dtype=torch.float32, device=device)
    
    # Define standardized domain for collocation points (all in [0,1])
    domain = {'x': (0.0, 1.0), 'z': (0.0, 1.0), 't': (0.0, 1.0)}
    
    # ----------------- Model Initialization -----------------
    model = PINN(input_dim=3, output_dim=3, hidden_layers=args.hidden_layers, hidden_size=args.hidden_size,
                 activation=args.activation, omega_0=args.omega_0,
                 use_fourier_features=args.use_fourier_features, mapping_size=args.mapping_size, fourier_scale=args.fourier_scale).to(device)
    
    # ----------------- Optimizer Setup -----------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Add learning rate scheduler (optional)
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=500, verbose=True
        )
    
    # ----------------- Training -----------------
    print("Training PINN...")
    loss_history = train_pinn(model, optimizer, x_data, measured_vel, domain,
                             n_colloc=args.n_colloc,
                             beta=args.beta, epochs=args.epochs, 
                             print_every=args.print_every,
                             scheduler=scheduler, adaptive_loss=args.adaptive_loss)
    print("Training complete!")
    print(f"Trainable viscosity mu: {model.get_mu().item():.4e}")
    
    # (Remaining processing: track extraction, interpolation, CSV writing, etc.)
    tracks = extract_tracks(x_data_np, time_round=2, distance_threshold=args.distance_threshold)
    print(f"Extracted {len(tracks)} raw tracks")
    interp_tracks = interpolate_tracks(tracks, interp_factor=args.interp_factor, smooth_factor=args.smooth_factor)
    print(f"Created {len(interp_tracks)} interpolated tracks")
    unnorm_tracks = unnormalize_tracks(interp_tracks, orig_bounds)
    write_tracks_csv(unnorm_tracks, args.tracks_csv, args.min_length, include_velocity=args.include_velocity)

# -----------------------------
# Argument Parsing and Script Execution
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ULM Pipeline PINN Framework")
    parser.add_argument("--csv_file", type=str, default="/Users/eric/Desktop/ULM_PI-LSTM/PINN/PINN Workspace/PALA_InVivoRatBrain_Coordinates.csv", 
                        help="Absolute path to microbubble tracking CSV file")
    parser.add_argument("--hidden_layers", type=int, default=5, help="Number of hidden layers in the network")
    parser.add_argument("--hidden_size", type=int, default=64, help="Number of neurons per hidden layer")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--n_colloc", type=int, default=1000, help="Number of collocation points")
    parser.add_argument("--beta", type=float, default=1000.0, help="Weight for the physics loss")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument("--print_every", type=int, default=250, help="Frequency of printing training info")
    parser.add_argument("--min_length", type=int, default=0, help="Minimal length of tracks to be saved")
    parser.add_argument("--tracks_csv", type=str, default="tracks.csv", help="Filename for the output tracks CSV file")
    parser.add_argument("--velocity_clip", type=float, default=0.1, help="Clipping value for normalized velocities")
    parser.add_argument("--interp_factor", type=float, default=5.0, help="Interpolation factor for tracks")
    parser.add_argument("--smooth_factor", type=int, default=20, help="Smoothing window size for track interpolation")
    parser.add_argument("--distance_threshold", type=float, default=0.05, help="Maximum distance for linking points in tracking")
    parser.add_argument("--include_velocity", action="store_true", help="Include velocity in the output CSV")
    parser.add_argument("--activation", type=str, default="tanh", help="Activation function for the network")
    parser.add_argument("--omega_0", type=float, default=30, help="Scaling factor for the first layer in sine activation")
    parser.add_argument("--use_fourier_features", action="store_true", default=False, help="Use Fourier feature mapping for input encoding")
    parser.add_argument("--mapping_size", type=int, default=32, help="Number of Fourier features for input encoding")
    parser.add_argument("--fourier_scale", type=float, default=5.0, help="Scaling factor for Fourier features")
    parser.add_argument("--use_scheduler", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--adaptive_loss", dest="adaptive_loss", action="store_true", default=True, help="Turn on adaptive loss function (default True)")
    parser.add_argument("--no-adaptive_loss", dest="adaptive_loss", action="store_false", help="Turn off adaptive loss function")
    args = parser.parse_args()
    
    main(args)
