import os
import sys
import yaml
import torch
import math
import numpy as np
import matplotlib.pyplot as plt

# Ensure the project directory is on PYTHONPATH
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from common.training import CheckpointLoader
from model import FLRONetFNO, FLRONetAFNO, FLRONetUNet, FLRONetMLP, FNO3D, FLRONetTransolver, FNO, AFNO, Transolver, UNet

def plot_heatmap(attn_matrix, out_path):
    """
    Plots a scientific cross-attention heatmap without captions, transparent background, and keeping grid.
    """
    h, w = attn_matrix.shape
    figwidth = 5.0
    figheight = figwidth * (h / w)
    fig, ax = plt.subplots(figsize=(figwidth, figheight), dpi=300)
    
    # Use RdYlBu_r for a vibrant blue-to-orange-red colormap
    im = ax.imshow(
        attn_matrix, 
        cmap='RdYlBu_r', 
        aspect='equal', 
        origin='upper',
        vmin=None,
        vmax=None
    )
    
    # Set major ticks but clear major labels and tick marks
    ax.set_xticks(np.arange(w))
    ax.set_yticks(np.arange(h))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', size=0)
    
    # Set minor ticks exactly between pixels to draw grid lines
    ax.set_xticks(np.arange(w + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(h + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(axis='both', which='minor', size=0) # Hide minor tick marks
    
    # Set spines (borders) to black with thin linewidth to match the grid
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.5)
        
    # Make background transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    print(f"Saved transparent heatmap to: {out_path}")

def main():
    # Load config.yaml
    config_path = os.path.join(PROJECT_DIR, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    checkpoint_path = os.path.join(PROJECT_DIR, config['evaluate']['from_checkpoint'])
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load model
    checkpoint_loader = CheckpointLoader(checkpoint_path=checkpoint_path)
    net = checkpoint_loader.load(scope=globals())
    net = net.cuda()
    net.eval()

    print(f"Loaded {net.__class__.__name__} successfully!")

    # Output directory
    out_dir = os.path.join(PROJECT_DIR, "plots_paper")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Generate 20:5 heatmaps
    sensor_times_20_5 = [0.0, 5.0, 10.0, 15.0, 20.0]
    target_times_20_5 = [float(t) for t in range(20)]
    
    sensor_t_20_5 = torch.tensor([sensor_times_20_5], dtype=torch.float, device="cuda")
    target_t_20_5 = torch.tensor([target_times_20_5], dtype=torch.float, device="cuda")

    with torch.no_grad():
        fullstate_emb_20_5 = net.sinusoid_embedding(target_t_20_5)
        sensor_emb_20_5 = net.sinusoid_embedding(sensor_t_20_5)
        trunk_outputs_20_5 = net.trunk_net(fullstate_emb_20_5, sensor_emb_20_5)
        
        # TrunkNet Stack 0
        attn_20_5 = trunk_outputs_20_5[0][0].cpu().numpy()
        
        # MeanField Weights
        if net.mean_field_net is not None:
            w_20_5 = net.mean_field_net._temporal_weights(sensor_t_20_5, target_t_20_5)
            w_np_20_5 = w_20_5[0].cpu().numpy()
        else:
            w_np_20_5 = None

    # Plot 20:5 TrunkNet Heatmap
    plot_heatmap(attn_20_5, os.path.join(out_dir, "attention_heatmap_20_5.png"))
    # Plot 20:5 MeanField Heatmap
    if w_np_20_5 is not None:
        plot_heatmap(w_np_20_5, os.path.join(out_dir, "meanfield_heatmap_20_5.png"))

    # 2. Generate 20:20 heatmaps
    sensor_times_20_20 = [float(t) for t in range(20)]
    target_times_20_20 = [float(t) for t in range(20)]
    
    sensor_t_20_20 = torch.tensor([sensor_times_20_20], dtype=torch.float, device="cuda")
    target_t_20_20 = torch.tensor([target_times_20_20], dtype=torch.float, device="cuda")

    with torch.no_grad():
        fullstate_emb_20_20 = net.sinusoid_embedding(target_t_20_20)
        sensor_emb_20_20 = net.sinusoid_embedding(sensor_t_20_20)
        trunk_outputs_20_20 = net.trunk_net(fullstate_emb_20_20, sensor_emb_20_20)
        
        # TrunkNet Stack 0
        attn_20_20 = trunk_outputs_20_20[0][0].cpu().numpy()
        
        # MeanField Weights
        if net.mean_field_net is not None:
            w_20_20 = net.mean_field_net._temporal_weights(sensor_t_20_20, target_t_20_20)
            w_np_20_20 = w_20_20[0].cpu().numpy()
        else:
            w_np_20_20 = None

    # Plot 20:20 TrunkNet Heatmap
    plot_heatmap(attn_20_20, os.path.join(out_dir, "attention_heatmap_20_20.png"))
    # Plot 20:20 MeanField Heatmap
    if w_np_20_20 is not None:
        plot_heatmap(w_np_20_20, os.path.join(out_dir, "meanfield_heatmap_20_20.png"))

    print("All 4 heatmaps successfully generated!")

if __name__ == "__main__":
    main()
