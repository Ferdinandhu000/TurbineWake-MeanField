"""
create_spliced_video.py  —  Generate spliced video and GIF from bin1 (last 50 frames) and bin2 (first 50 frames)
=========================================================================================================

Reads:
  - data/case5_lowTi_UWV_10min_bin1.nc
  - data/case5_lowTi_UWV_10min_bin2.nc

Visualizes:
  - U component velocity field on cropped downstream domain (120x256)
  - Seamlessly joins bin1 (last 50 frames, index 2450..2499) and bin2 (first 50 frames, index 0..49)
  - Outputs an MP4 video and an animated GIF

Aesthetics:
  - Uses beautiful viridis/jet colormap with coordinate labels normalized by rotor diameter D = 126m
  - Annotates current frame source, index, and timestamp
  - Draws a clear highlight indicator when transitioning from bin1 to bin2
"""

import os
import shutil
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# ── configuration ─────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), 'data')
OUT_MP4     = os.path.join(os.path.dirname(__file__), 'spliced_wake_flow.mp4')
OUT_GIF     = os.path.join(os.path.dirname(__file__), 'spliced_wake_flow.gif')

# Artifact folder for embedding in walkthrough
ARTIFACT_DIR = r"C:\Users\HJ000\.gemini\antigravity\brain\4f8d4484-bd44-4367-93ce-ed35ad8e59de"

NX_ORIGINAL = 286
NY_ORIGINAL = 121
X_START_IDX = 30      # downstream domain start
Y_END_IDX   = 120     # crop y-dimension
D           = 126.0   # rotor diameter

def main():
    bin1_path = os.path.join(DATA_DIR, 'case5_lowTi_UWV_10min_bin1.nc')
    bin2_path = os.path.join(DATA_DIR, 'case5_lowTi_UWV_10min_bin2.nc')

    print("Loading datasets...")
    ds1 = xr.open_dataset(bin1_path)
    ds2 = xr.open_dataset(bin2_path)

    # Coordinates in rotor diameter units
    x_coords = ds1['x'].values[X_START_IDX:] / D
    y_coords = ds1['y'].values[:Y_END_IDX] / D

    print("Extracting last 50 frames of U from bin1...")
    # shape (50, nx, ny)
    u_bin1 = ds1['U'].values[-50:, X_START_IDX:, :Y_END_IDX]
    print("Extracting first 50 frames of U from bin2...")
    u_bin2 = ds2['U'].values[:50, X_START_IDX:, :Y_END_IDX]

    # Transpose to shape (T, H, W) = (T, 120, 256)
    u_bin1 = u_bin1.transpose(0, 2, 1)
    u_bin2 = u_bin2.transpose(0, 2, 1)

    # Close datasets
    ds1.close()
    ds2.close()

    # Combine frames
    u_combined = np.concatenate([u_bin1, u_bin2], axis=0)
    nt_combined = len(u_combined)
    print(f"Combined data shape: {u_combined.shape}")

    # Matplotlib styling
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.12)

    # Initialize plot
    vmin, vmax = 1.0, 10.5
    im = ax.imshow(
        u_combined[0],
        extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
        origin='lower',
        cmap='jet',
        vmin=vmin,
        vmax=vmax,
        aspect='equal'
    )
    cb = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.18, shrink=0.7)
    cb.set_label('U Velocity Component (m/s)', fontsize=10, fontweight='bold')

    ax.set_xlim(x_coords[0], x_coords[-1])
    ax.set_ylim(y_coords[0], y_coords[-1])
    ax.set_xlabel('$x/D$ (Downstream Distance)', fontsize=10, fontweight='bold')
    ax.set_ylabel('$y/D$ (Spanwise Distance)', fontsize=10, fontweight='bold')
    title_text = fig.suptitle('Spliced Wake Velocity Field: case5_lowTi', fontsize=12, fontweight='bold')

    # Text annotations on plot
    source_text = ax.text(0.02, 0.93, '', transform=ax.transAxes, color='white', fontsize=10,
                          weight='bold', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
    
    frame_idx_text = ax.text(0.85, 0.93, '', transform=ax.transAxes, color='white', fontsize=10,
                             weight='bold', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))

    # Temporary list to store image arrays
    frames_bgr = []
    pil_images = []

    print("Generating frames...")
    for t in range(nt_combined):
        # Update flow data
        im.set_data(u_combined[t])

        # Update labels and details
        if t < 50:
            source_file = "bin1 (last 50 frames)"
            frame_num = 2450 + t
            border_color = 'white'
        else:
            source_file = "bin2 (first 50 frames)"
            frame_num = t - 50
            border_color = 'cyan'

        source_text.set_text(f"Source: {source_file}")
        source_text.set_bbox(dict(facecolor='black', edgecolor=border_color, alpha=0.7, boxstyle='round,pad=0.3'))
        frame_idx_text.set_text(f"Frame Index: {frame_num:04d}")

        # Red raw-line transition indicator flashing near t=50
        if 48 <= t <= 52:
            ax.set_title("--- SPLICING JUNCTION AREA ---", color='red', fontweight='bold', fontsize=11)
        else:
            ax.set_title("Convective Wake Profile Flow", color='black', fontsize=10)

        # Render
        fig.canvas.draw()

        # Convert canvas to image
        rgba_buffer = fig.canvas.buffer_rgba()
        frame_rgba = np.asarray(rgba_buffer)
        
        # Keep RGB copy for PIL (GIF)
        frame_rgb = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2RGB)
        pil_images.append(Image.fromarray(frame_rgb))

        # Convert to BGR for OpenCV (MP4)
        frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
        frames_bgr.append(frame_bgr)

    plt.close(fig)

    # 1. Write MP4 video using OpenCV
    print(f"Writing MP4 video to {OUT_MP4} ...")
    height, width, _ = frames_bgr[0].shape
    # Use MP4V codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUT_MP4, fourcc, 10.0, (width, height))  # 10 fps (10s total)
    for frame in frames_bgr:
        video_writer.write(frame)
    video_writer.release()
    print("MP4 video saved successfully.")

    # 2. Write Animated GIF using PIL
    print(f"Writing Animated GIF to {OUT_GIF} ...")
    # Save GIF, duration=100ms per frame (10 fps), loop infinitely (loop=0)
    pil_images[0].save(
        OUT_GIF,
        save_all=True,
        append_images=pil_images[1:],
        duration=100,
        loop=0
    )
    print("GIF saved successfully.")

    # Copy files to Artifact directory for display in Walkthrough if exists
    if os.path.exists(ARTIFACT_DIR):
        artifact_mp4_path = os.path.join(ARTIFACT_DIR, 'spliced_wake_flow.mp4')
        artifact_gif_path = os.path.join(ARTIFACT_DIR, 'spliced_wake_flow.gif')
        print(f"Copying video files to artifact folder: {ARTIFACT_DIR} ...")
        shutil.copy(OUT_MP4, artifact_mp4_path)
        shutil.copy(OUT_GIF, artifact_gif_path)
        print("Copied files to artifact directory successfully.")

if __name__ == '__main__':
    main()
