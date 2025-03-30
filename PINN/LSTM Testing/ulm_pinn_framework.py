"""
ULM PINN Tracking Framework
---------------------------
This script implements a physics-informed neural network (PINN) to post process
microbubble localization data (loaded from "mb_ulm_4con.csv") and reconstruct
a dense velocity and pressure fields while simultaneously providing improved
tracking.
It also features two added functions:
  (i) visual output (overlaid bubble trajectories compared to predicted velocity field)
  (ii) an option to generate a PDF report of figures.
  
Author: Your Name
Date: YYYY-MM-DD
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 1. Data Preprocessing Functions
# -----------------------------
def load_tracking_data(csv_file):
    """
    Load the microbubble tracking CSV file.
    Expected columns: Intensity, X, Y, ImageIndex.
    For demonstration, we normalize coordinates and assign time = ImageIndex.
    Returns
        X: np.array of shape (N,)
        Y: np.array of shape (N,)
        T: np.array of shape (N,)  (converted from ImageIndex)
        measured_vel: np.array of shape (N,2) of finite-difference velocities.
                      (Here we use a dummy velocity for demonstration.)
    """
    df = pd.read_csv(csv_file)
    # Normalize spatial coordinates (example normalization: min-max)
    X = df['X'].values.astype(np.float32)
    Y = df['Y'].values.astype(np.float32)
    T = df['ImageIndex'].values.astype(np.float32)  # time can be from frame index

    # For a real application: one would group detections to form trajectories.
    # Here we simply approximate the measured velocity by finite difference along sorted time.
    # We sort by time:
    sort_idx = np.argsort(T)
    X, Y, T = X[sort_idx], Y[sort_idx], T[sort_idx]
    # Approximate velocity by using central difference (pad with zeros at boundaries)
    u_meas = np.zeros_like(X)
    v_meas = np.zeros_like(Y)
    if len(T) > 2:
        dt = np.diff(T)
        # For internal indices (central difference)
        u_meas[1:-1] = (X[2:] - X[:-2]) / (T[2:] - T[:-2] + 1e-6)
        v_meas[1:-1] = (Y[2:] - Y[:-2]) / (T[2:] - T[:-2] + 1e-6)
    measured_vel = np.stack([u_meas, v_meas], axis=-1)
    
    # Normalize coordinates (optional, here we scale into [0,1])
    X_norm = (X - X.min()) / (X.max()-X.min() + 1e-6)
    Y_norm = (Y - Y.min()) / (Y.max()-Y.min() + 1e-6)
    T_norm = (T - T.min()) / (T.max()-T.min() + 1e-6)
    
    return X_norm, Y_norm, T_norm, measured_vel

def generate_collocation_points(n_points, domain):
    """
    Generate random collocation points (for physics loss) in the domain.
    Domain is a dict with keys: 'x', 'y', 't', each a tuple: (min, max).
    Returns: numpy array of shape (n_points, 3) 
             where each row is (x,y,t).
    """
    x = np.random.uniform(domain['x'][0], domain['x'][1], n_points)
    y = np.random.uniform(domain['y'][0], domain['y'][1], n_points)
    t = np.random.uniform(domain['t'][0], domain['t'][1], n_points)
    return np.stack([x,y,t], axis=-1)

# -----------------------------
# 2. PINN Network Definition
# -----------------------------
class PINN(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_layers=8, hidden_size=50):
        """
        A fully connected deep network with tanh activations.
        Input: (x, y, t)
        Output: (u, v, p)
        """
        super(PINN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers-1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Trainable log-viscosity parameter
        self.log_mu = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        
    def forward(self, x):
        # x has shape (N,3) : columns x, y, t.
        out = self.network(x)
        # u = out[:,0], v = out[:,1], p = out[:,2]
        u = out[:,0:1]
        v = out[:,1:2]
        p = out[:,2:3]
        return u, v, p
    
    def get_mu(self):
        return torch.exp(self.log_mu)

# -----------------------------
# 3. Loss Functions
# -----------------------------
def data_loss(model, x_data, measured_vel):
    """
    Compute mean squared error between measured velocity (from tracking data)
    and the PINN predicted velocity.
    Inputs:
       model: the PINN model
       x_data: tensor (N,3) of measured coordinates (x,y,t) with requires_grad False.
       measured_vel: tensor of shape (N,2) with measured [u,v].
    """
    u_pred, v_pred, _ = model(x_data)
    pred_vel = torch.cat((u_pred, v_pred), dim=1)
    loss = torch.mean((pred_vel - measured_vel)**2)
    return loss

def physics_loss(model, x_colloc):
    """
    Compute the physics residual by automatic differentiation.
    Enforce:
       - Momentum residual (for u and v components):
           R_u = -p_x + mu*(u_xx + u_yy) = 0
           R_v = -p_y + mu*(v_xx + v_yy) = 0
       - Mass conservation: u_x + v_y = 0
    x_colloc: tensor (N,3) with requires_grad=True.
    """
    x_colloc.requires_grad_(True)
    u, v, p = model(x_colloc)
    mu = model.get_mu()
    
    # First derivatives
    grads = {}
    for var, name in zip([u, v, p], ['u','v','p']):
        grad = torch.autograd.grad(var, x_colloc, grad_outputs=torch.ones_like(var), 
                             retain_graph=True, create_graph=True)[0]
        grads[name] = grad  # shape (N,3) ; columns correspond to d/dx, d/dy, d/dt
    
    u_x = grads['u'][:,0:1]
    u_y = grads['u'][:,1:2]
    
    v_x = grads['v'][:,0:1]
    v_y = grads['v'][:,1:2]
    
    p_x = grads['p'][:,0:1]
    p_y = grads['p'][:,1:2]
    
    # Laplacian for u (second derivatives)
    u_xx = torch.autograd.grad(u_x, x_colloc, grad_outputs=torch.ones_like(u_x), 
                               retain_graph=True, create_graph=True)[0][:,0:1]
    u_yy = torch.autograd.grad(u_y, x_colloc, grad_outputs=torch.ones_like(u_y), 
                               retain_graph=True, create_graph=True)[0][:,1:2]
    
    # Laplacian for v (second derivatives)
    v_xx = torch.autograd.grad(v_x, x_colloc, grad_outputs=torch.ones_like(v_x),
                               retain_graph=True, create_graph=True)[0][:,0:1]
    v_yy = torch.autograd.grad(v_y, x_colloc, grad_outputs=torch.ones_like(v_y),
                               retain_graph=True, create_graph=True)[0][:,1:2]
    
    # Residuals
    R_u = -p_x + mu * (u_xx + u_yy)
    R_v = -p_y + mu * (v_xx + v_yy)
    R_div = u_x + v_y
    
    loss_phys = torch.mean(R_u**2) + torch.mean(R_v**2) + torch.mean(R_div**2)
    return loss_phys

# -----------------------------
# 4. Training Function
# -----------------------------
def train_pinn(model, optimizer, x_data, measured_vel, domain,
               n_colloc=1000, beta=1.0, epochs=5000, print_every=500):
    """
    Main training loop.
    - x_data: measured data (tensor, shape (N,3))
    - measured_vel: measured velocities (tensor, shape (N,2))
    - domain: dictionary defining the limits for collocation points.
    - beta: scaling factor for the physics loss.
    """
    model.train()
    loss_history = []
    for ep in range(1, epochs+1):
        optimizer.zero_grad()
        # Data loss
        loss_d = data_loss(model, x_data, measured_vel)
        
        # Generate collocation points (randomly sample at each epoch)
        colloc_np = generate_collocation_points(n_colloc, domain)
        x_colloc = torch.tensor(colloc_np, dtype=torch.float32, device=x_data.device)
        
        loss_p = physics_loss(model, x_colloc)
        
        loss_total = loss_d + beta * loss_p
        loss_total.backward()
        optimizer.step()
        
        loss_history.append(loss_total.item())
        if ep % print_every == 0:
            print(f"Epoch {ep:05d}: Loss_total={loss_total.item():.4e}, Data Loss={loss_d.item():.4e}, Phys Loss={loss_p.item():.4e}, mu={model.get_mu().item():.4e}")
    return loss_history

# -----------------------------
# 5. Visualization Functions
# -----------------------------
def plot_velocity_field(model, domain, n_grid=20, t_val=0.5, ax=None):
    """
    Plot the predicted velocity field (quiver) for a fixed time t=t_val.
    Domain: dictionary with keys 'x','y'
    """
    x = np.linspace(domain['x'][0], domain['x'][1], n_grid)
    y = np.linspace(domain['y'][0], domain['y'][1], n_grid)
    X, Y = np.meshgrid(x,y)
    T = t_val*np.ones_like(X)
    
    grid = np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=-1)
    with torch.no_grad():
        grid_tensor = torch.tensor(grid, dtype=torch.float32)
        u_pred, v_pred, _ = model(grid_tensor)
        u_np = u_pred.cpu().numpy().reshape(X.shape)
        v_np = v_pred.cpu().numpy().reshape(Y.shape)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    ax.quiver(X, Y, u_np, v_np)
    ax.set_title(f"Predicted Velocity Field at t={t_val:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax

def plot_tracking_results(x_data, measured_vel, model, ax=None):
    """
    Plot the original bubble positions (from x_data) and compare with the integrated velocity field.
    For demonstration we overlay scatter data with the predicted field.
    """
    # x_data is tensor with columns [x,y,t]. For simplicity, we plot only those with a given time slice.
    x_np = x_data.cpu().numpy()
    t_mean = np.median(x_np[:,2])
    # Select points with time near median
    tol = 0.05
    idx = np.where(np.abs(x_np[:,2] - t_mean)<tol)[0]
    if len(idx)==0:
        idx = np.arange(len(x_np))
    pts = x_np[idx]
    # Predicted velocities at these points
    with torch.no_grad():
        pts_tensor = torch.tensor(pts, dtype=torch.float32)
        u_pred, v_pred, _ = model(pts_tensor)
        u_pred = u_pred.cpu().numpy().flatten()
        v_pred = v_pred.cpu().numpy().flatten()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    # Scatter measured positions
    ax.scatter(pts[:,0], pts[:,1], color='blue', label='Measured positions', s=10)
    # Overlay quiver from predicted velocities
    ax.quiver(pts[:,0], pts[:,1], u_pred, v_pred, color='red', scale=1, scale_units='xy', angles='xy', label='Predicted velocity')
    ax.set_title(f"Tracking Results at t ~ {t_mean:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    return ax

def compute_l2_error(model, x_data, measured_vel):
    """
    Compute the L2 norm error at the measured data points.
    """
    with torch.no_grad():
        u_pred, v_pred, _ = model(x_data)
        pred_vel = torch.cat((u_pred, v_pred), dim=1)
        error = torch.norm(pred_vel - measured_vel, p=2) / torch.norm(measured_vel, p=2)
    return error.item()

# -----------------------------
# 6. PDF Report Export
# -----------------------------
def generate_pdf_report(model, x_data, measured_vel, domain, pdf_filename="tracking_report.pdf"):
    """
    Generate a PDF report containing:
      - The velocity field quiver plot
      - The tracking results plot
      - A text page with error metrics and training details.
    """
    pp = PdfPages(pdf_filename)
    
    # Figure 1: Velocity field at fixed t.
    fig1, ax1 = plt.subplots(figsize=(6,5))
    plot_velocity_field(model, domain, n_grid=20, t_val=0.5, ax=ax1)
    fig1.suptitle("Dense Velocity Field Prediction")
    pp.savefig(fig1)
    plt.close(fig1)
    
    # Figure 2: Tracking results% (overlay)
    fig2, ax2 = plt.subplots(figsize=(6,5))
    plot_tracking_results(x_data, measured_vel, model, ax=ax2)
    fig2.suptitle("Overlay: Measured Bubble Positions and Predicted Velocities")
    pp.savefig(fig2)
    plt.close(fig2)
    
    # Figure 3: Error metrics text page
    err = compute_l2_error(model, x_data, measured_vel)
    fig3 = plt.figure(figsize=(8,6))
    plt.axis('off')
    txt = f"Final L2 Relative Error between measured and predicted velocities: {err:.4e}\n"
    txt += f"Trainable viscosity mu: {model.get_mu().item():.4e}\n"
    plt.text(0.1, 0.5, txt, fontsize=14)
    pp.savefig(fig3)
    plt.close(fig3)
    
    pp.close()
    print(f"PDF report saved to {pdf_filename}")

# -----------------------------
# 7. Main Function and Argument Parsing
# -----------------------------
def main(args):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # Load tracking data
    csv_file = args.csv_file
    X, Y, T, measured_vel_np = load_tracking_data(csv_file)
    print(f"Loaded {len(X)} data points.")
    
    # Create tensors from data
    x_data_np = np.stack([X, Y, T], axis=-1)
    x_data = torch.tensor(x_data_np, dtype=torch.float32, device=device)
    measured_vel = torch.tensor(measured_vel_np, dtype=torch.float32, device=device)
    
    # Determine domain from data (or set manually)
    domain = {
        'x': (0.0, 1.0),
        'y': (0.0, 1.0),
        't': (0.0, 1.0)
    }
    
    # Initialize model and optimizer
    model = PINN(input_dim=3, output_dim=3, hidden_layers=args.hidden_layers, hidden_size=args.hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("Training PINN...")
    loss_history = train_pinn(model, optimizer, x_data, measured_vel, domain,
                              n_colloc=args.n_colloc, beta=args.beta, epochs=args.epochs, print_every=args.print_every)
    print("Training complete!")
    
    # Compute error metric on training data
    err = compute_l2_error(model, x_data, measured_vel)
    print(f"Final L2 relative error: {err:.4e}")
    
    # Plot and show figures
    fig, ax = plt.subplots(figsize=(6,5))
    plot_velocity_field(model, domain, n_grid=20, t_val=0.5, ax=ax)
    plt.show()
    
    fig, ax = plt.subplots(figsize=(6,5))
    plot_tracking_results(x_data, measured_vel, model, ax=ax)
    plt.show()
    
    # Optionally, save the report as a PDF
    if args.save_pdf:
        generate_pdf_report(model, x_data, measured_vel, domain, pdf_filename=args.pdf_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ULM Pipeline PINN Framework")
    parser.add_argument("--csv_file", type=str, default="mb_ulm_4con.csv", help="Path to microbubble tracking CSV file")
    parser.add_argument("--hidden_layers", type=int, default=8, help="Number of hidden layers in the network")
    parser.add_argument("--hidden_size", type=int, default=50, help="Number of neurons per hidden layer")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--n_colloc", type=int, default=1000, help="Number of collocation points")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for the physics loss")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument("--print_every", type=int, default=500, help="Frequency of printing training info")
    parser.add_argument("--save_pdf", action="store_true", help="If set, generate a PDF report of results")
    parser.add_argument("--pdf_filename", type=str, default="tracking_report.pdf", help="Filename for the PDF report")
    args = parser.parse_args()
    
    main(args) 