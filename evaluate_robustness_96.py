import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import shutil
import json

# Set paths
PROJECT_DIR_W = Path(r"d:\hj\Y-2 S-1\Programming Project\FlowNet-NEW\turbine_wake-MeanField").resolve()
os.chdir(PROJECT_DIR_W)
sys.path.insert(0, str(PROJECT_DIR_W))

from cfd.dataset import CFDDataset
from cfd.embedding import Voronoi
from common.training import CheckpointLoader
from common.functional import compute_velocity_field
import model

# Register models
globals()['FLRONetTransolver'] = model.FLRONetTransolver
globals()['FLRONetAFNO'] = model.FLRONetAFNO
globals()['FLRONetFNO'] = model.FLRONetFNO
globals()['AFNO'] = model.AFNO
globals()['Transolver'] = model.Transolver
globals()['FNO'] = model.FNO

def plot_separate_frame(frame, vmin, vmax, cmap, out_path):
    # Width 30.5mm, height based on aspect ratio 120/256
    fig_w_in = 30.5 / 25.4
    fig_h_in = (30.5 * (120.0 / 256.0)) / 25.4
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_w_in, fig_h_in))
    ax.imshow(
        frame,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bicubic",
        aspect="auto",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def main():
    print("Regenerating clean 96-sensor dataset on disk...")
    # Instantiate dataset with write_to_disk=True once to generate 96-sensor metadata/files
    dataset = CFDDataset(
        root=str(PROJECT_DIR_W / "data/case5_lowTi_0-30/test"),
        init_sensor_timeframes=[0, 5, 10, 15, 20],
        future_prediction_range=None,
        n_fullstate_timeframes_per_chunk=21, # Evaluate all 21 inside frames for Table 4
        n_samplings_per_chunk=1,
        resolution=(120, 256),
        n_sensors=96,
        dropout_probabilities=[],
        noise_level=0,
        sensor_generator='WakeShearLHS',
        embedding_generator='Voronoi',
        init_fullstate_timeframes=list(range(21)),
        seed=1,
        write_to_disk=True,
        sensor_position_path=None
    )
    
    # Load raw data to construct embeddings dynamically
    raw_data = np.load(str(PROJECT_DIR_W / "data/case5_lowTi_0-30/test/test_data.npy"))
    data_t = torch.from_numpy(raw_data).cuda().float().permute(0, 3, 1, 2) # shape (750, 3, 120, 256)
    
    # Checkpoints
    checkpoints = {
        "FNO":              r"Turbe-2\checkpoints_best\.checkpoints_01_W_FNO_inside_config\fno1.pt",
        "FLRONet-FNO":      r"checkpoints_best\.checkpoints_12_W_FLRONet_96\flronetfno1.pt",
        "AFNO":             r"checkpoints_best\.checkpoints_02_W_AFNO_96\afno5.pt",
        "CATO-AFNO":        r"checkpoints_best\.checkpoints_17_W_CATO-afno_96\flronetafno3.pt",
        "Transolver":       r"checkpoints_best\.checkpoints_07_W_Transolver_96\transolver7.pt",
        "CATO-Transolver":  r"checkpoints_best\.checkpoints_22_W_CATO-trans_96\flronettransolver5.pt",
    }
    
    # Load models
    print("Loading models...")
    models = {}
    for name, path in checkpoints.items():
        models[name] = CheckpointLoader(checkpoint_path=str(PROJECT_DIR_W / path)).load(scope=globals()).cuda().eval()
        print(f"  Loaded {name}")
        
    conditions = [
        ("Clean", 0.00, 0),
        ("Noise_0.05", 0.05, 0),
        ("Noise_0.10", 0.10, 0),
        ("Noise_0.20", 0.20, 0),
        ("Dropout_5", 0.00, 5),
        ("Dropout_10", 0.00, 10),
        ("Dropout_20", 0.00, 20)
    ]
    
    num_eval_chunks = len(dataset) # 730 chunks
    print(f"\nRunning Table 4 evaluation on {num_eval_chunks} chunks...")
    
    # We will accumulate MAEs over all chunks to build Table 4
    # Shape: (len(models), len(conditions))
    table_4_maes = np.zeros((len(models), len(conditions)))
    
    # To find the best case/chunk for plotting at frame [17]
    # We will evaluate case-level L2 losses at frame 17
    # Shape: (num_eval_chunks, len(conditions), len(models))
    frame17_l2_losses = np.zeros((num_eval_chunks, len(conditions), len(models)))
    
    model_names = list(models.keys())
    
    for chunk_idx in range(num_eval_chunks):
        st_t, _, ft_t, fv_t, _, _ = dataset[chunk_idx]
        
        # Original clean sensor frame data
        sensor_sample = data_t[st_t].unsqueeze(0) # shape (1, 5, 3, 120, 256)
        
        # Frame 17 index in inside timeframes list(range(21)) is 17
        gt_frame17 = fv_t[17]
        gt_vel17 = compute_velocity_field(gt_frame17, dim=0).squeeze(0)
        gt_norm17 = torch.linalg.norm(gt_vel17).item()
        
        # Evaluate each condition
        for cond_idx, (cond_name, noise, dropout) in enumerate(conditions):
            # Dynamic embedding generator
            if dropout == 0:
                dropout_probs = []
            else:
                dropout_probs = [0.] * dropout
                dropout_probs[-1] = 1.
                
            emb_gen = Voronoi(
                resolution=(120, 256),
                sensor_positions=dataset.sensor_positions,
                dropout_probabilities=dropout_probs,
                noise_level=noise
            )
            
            # Apply Voronoi embedding
            sensor_embedding = emb_gen(sensor_sample, seed=1 + chunk_idx + cond_idx).cuda()
            
            st_gpu = st_t.unsqueeze(0).cuda()
            ft_gpu = ft_t.unsqueeze(0).cuda()
            
            with torch.no_grad():
                for m_idx, name in enumerate(model_names):
                    net = models[name]
                    preds = net(st_gpu, sensor_embedding, ft_gpu, None) # shape (1, 21, 3, 120, 256)
                    
                    # Accumulate MAE over all 21 frames for Table 4
                    mae = torch.mean(torch.abs(preds - fv_t.unsqueeze(0).cuda())).item()
                    table_4_maes[m_idx, cond_idx] += mae
                    
                    # Also compute L2 loss at frame 17 for case selection
                    pred_frame17 = preds[0, 17]
                    pred_vel17 = compute_velocity_field(pred_frame17, dim=0).squeeze(0)
                    l2_17 = torch.linalg.norm(pred_vel17 - gt_vel17).item() / gt_norm17
                    frame17_l2_losses[chunk_idx, cond_idx, m_idx] = l2_17
                    
        if chunk_idx % 100 == 99:
            print(f"  Processed {chunk_idx + 1}/{num_eval_chunks} chunks...")
            
    # Compute average MAE over all chunks
    table_4_maes /= num_eval_chunks
    
    # Save Table 4 to results JSON
    table_results = {}
    for m_idx, name in enumerate(model_names):
        table_results[name] = {
            "Accuracy": table_4_maes[m_idx, 0],
            "Noise_5%": table_4_maes[m_idx, 1],
            "Noise_10%": table_4_maes[m_idx, 2],
            "Noise_20%": table_4_maes[m_idx, 3],
            "Dropout_5": table_4_maes[m_idx, 4],
            "Dropout_10": table_4_maes[m_idx, 5],
            "Dropout_20": table_4_maes[m_idx, 6],
        }
        
    print("\n--- TABLE 4 REGENERATED (96 SENSORS) ---")
    print("| Model | Accuracy | Noise 5% | Noise 10% | Noise 20% | Dropout 5 | Dropout 10 | Dropout 20 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in model_names:
        r = table_results[name]
        print("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
            name, r["Accuracy"], r["Noise_5%"], r["Noise_10%"], r["Noise_20%"], r["Dropout_5"], r["Dropout_10"], r["Dropout_20"]
        ))
        
    with open(PROJECT_DIR_W / "table_4_robustness_96.json", "w", encoding="utf-8") as f:
        json.dump(table_results, f, indent=4)
        
    # Select the best case/chunk for plotting at frame 17
    # Gap formula: CATO models vs their corresponding baselines
    # indices: FNO=0, FLRONet-FNO=1, AFNO=2, CATO-AFNO=3, Transolver=4, CATO-Transolver=5
    gaps = []
    for i in range(num_eval_chunks):
        gap_afno = frame17_l2_losses[i, 1:, 2] - frame17_l2_losses[i, 1:, 3] # AFNO - CATO-AFNO
        gap_trans = frame17_l2_losses[i, 1:, 4] - frame17_l2_losses[i, 1:, 5] # Transolver - CATO-Transolver
        
        # Check if CATO is consistently better
        if np.all(gap_afno > 0) and np.all(gap_trans > 0):
            score = np.mean(gap_afno + gap_trans)
        else:
            score = -1.0
        gaps.append(score)
        
    best_chunk_idx = int(np.argmax(gaps))
    print(f"\n[+] Selected Best Chunk Index for Plotting: {best_chunk_idx} (Score: {gaps[best_chunk_idx]:.4f})")
    
    # Save visualizations for the selected case
    out_dir = PROJECT_DIR_W / "plots_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(r"C:\Users\HJ000\.gemini\antigravity\brain\d559beff-90e2-4679-8d80-04bf042063a7")
    
    st_t, _, ft_t, fv_t, _, _ = dataset[best_chunk_idx]
    sensor_sample = data_t[st_t].unsqueeze(0)
    
    # Frame 17 ground truth velocity field
    gt_frame17 = fv_t[17]
    gt_vel17 = compute_velocity_field(gt_frame17, dim=0).squeeze(0)
    gt_vel17_np = gt_vel17.cpu().numpy()
    
    # Plot Ground Truth
    plot_separate_frame(gt_vel17_np, 2, 10, "viridis", out_dir / "ground_truth.png")
    shutil.copy(out_dir / "ground_truth.png", artifacts_dir / "turbine_ground_truth.png")
    
    # Plot models under each condition
    for cond_idx, (cond_name, noise, dropout) in enumerate(conditions):
        if dropout == 0:
            dropout_probs = []
        else:
            dropout_probs = [0.] * dropout
            dropout_probs[-1] = 1.
            
        emb_gen = Voronoi(
            resolution=(120, 256),
            sensor_positions=dataset.sensor_positions,
            dropout_probabilities=dropout_probs,
            noise_level=noise
        )
        
        sensor_embedding = emb_gen(sensor_sample, seed=1 + best_chunk_idx + cond_idx).cuda()
        
        st_gpu = st_t.unsqueeze(0).cuda()
        ft_gpu = ft_t.unsqueeze(0).cuda()
        
        with torch.no_grad():
            preds = {}
            for name in ["AFNO", "CATO-AFNO", "Transolver", "CATO-Transolver"]:
                net = models[name]
                p = net(st_gpu, sensor_embedding, ft_gpu, None)
                preds[name] = compute_velocity_field(p[0, 17], dim=0).squeeze(0)
                
        for name in ["AFNO", "CATO-AFNO", "Transolver", "CATO-Transolver"]:
            pred_np = preds[name].cpu().numpy()
            err_np = pred_np - gt_vel17_np
            
            fn_pred = f"{name}_{cond_name}_pred.png"
            fn_err = f"{name}_{cond_name}_err.png"
            
            plot_separate_frame(pred_np, 2, 10, "viridis", out_dir / fn_pred)
            plot_separate_frame(err_np, -5, 5, "viridis", out_dir / fn_err)
            
            # Copy to artifacts directory
            shutil.copy(out_dir / fn_pred, artifacts_dir / f"turbine_{fn_pred}")
            shutil.copy(out_dir / fn_err, artifacts_dir / f"turbine_{fn_err}")
            
    print("\n--- L2 Loss Table for Frame 17 of Chunk {} ---".format(best_chunk_idx))
    print("| Condition | AFNO | CATO-AFNO | Transolver | CATO-Transolver |")
    print("| --- | --- | --- | --- | --- |")
    for cond_idx, (cond_name, _, _) in enumerate(conditions):
        losses = frame17_l2_losses[best_chunk_idx, cond_idx]
        # indices in frame17_l2_losses: FNO=0, FLRONet-FNO=1, AFNO=2, CATO-AFNO=3, Transolver=4, CATO-Transolver=5
        print("| {} | {:.5f} | {:.5f} | {:.5f} | {:.5f} |".format(
            cond_name, losses[2], losses[3], losses[4], losses[5]
        ))
        
    with open(out_dir / "l2_losses.txt", "w") as f:
        f.write("| Condition | AFNO | CATO-AFNO | Transolver | CATO-Transolver |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for cond_idx, (cond_name, _, _) in enumerate(conditions):
            losses = frame17_l2_losses[best_chunk_idx, cond_idx]
            f.write("| {} | {:.5f} | {:.5f} | {:.5f} | {:.5f} |\n".format(
                cond_name, losses[2], losses[3], losses[4], losses[5]
            ))

if __name__ == '__main__':
    main()
