import os
import sys
from pathlib import Path
import torch
import numpy as np
import json

# Set paths
PROJECT_DIR_W = Path(r"d:\hj\Y-2 S-1\Programming Project\FlowNet-NEW\turbine_wake-MeanField").resolve()
os.chdir(PROJECT_DIR_W)
sys.path.insert(0, str(PROJECT_DIR_W))

from cfd.dataset import CFDDataset
from cfd.embedding import Voronoi
from common.training import CheckpointLoader
import model

# Register models
globals()['FLRONetTransolver'] = model.FLRONetTransolver
globals()['FLRONetAFNO'] = model.FLRONetAFNO
globals()['FLRONetFNO'] = model.FLRONetFNO
globals()['AFNO'] = model.AFNO
globals()['Transolver'] = model.Transolver
globals()['FNO'] = model.FNO

def main():
    print("Loading Turbine Wake dataset...")
    dataset = CFDDataset(
        root=str(PROJECT_DIR_W / "data/case5_lowTi_0-30/test"),
        init_sensor_timeframes=[0, 5, 10, 15, 20],
        future_prediction_range=None,
        n_fullstate_timeframes_per_chunk=21,
        n_samplings_per_chunk=1,
        resolution=(120, 256),
        n_sensors=96,
        dropout_probabilities=[],
        noise_level=0,
        sensor_generator='WakeShearLHS',
        embedding_generator='Voronoi',
        init_fullstate_timeframes=list(range(21)),
        seed=1,
        write_to_disk=False,
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
        
    conditions = [
        ("Clean", 0.00, 0),
        ("Noise_0.05", 0.05, 0),
        ("Noise_0.10", 0.10, 0),
        ("Noise_0.20", 0.20, 0),
        ("Dropout_5", 0.00, 5),
        ("Dropout_10", 0.00, 10),
        ("Dropout_20", 0.00, 20)
    ]
    
    num_eval_chunks = len(dataset)
    print(f"Running relative L2 loss robustness evaluation on {num_eval_chunks} chunks...")
    
    # Accumulate relative L2 losses
    # Shape: (len(models), len(conditions))
    table_4_l2_losses = np.zeros((len(models), len(conditions)))
    model_names = list(models.keys())
    
    for chunk_idx in range(num_eval_chunks):
        st_t, _, ft_t, fv_t, _, _ = dataset[chunk_idx]
        sensor_sample = data_t[st_t].unsqueeze(0) # shape (1, 5, 3, 120, 256)
        
        fv_gpu = fv_t.unsqueeze(0).cuda()
        
        # Precompute L2 norm of the ground truth for relative error
        # Shape: (1, 21, 3)
        gt_norm = torch.linalg.norm(fv_gpu.flatten(3), dim=-1)
        
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
            
            sensor_embedding = emb_gen(sensor_sample, seed=1 + chunk_idx + cond_idx).cuda()
            st_gpu = st_t.unsqueeze(0).cuda()
            ft_gpu = ft_t.unsqueeze(0).cuda()
            
            with torch.no_grad():
                for m_idx, name in enumerate(model_names):
                    net = models[name]
                    preds = net(st_gpu, sensor_embedding, ft_gpu, None) # shape (1, 21, 3, 120, 256)
                    
                    # Compute relative L2 error: norm(pred - gt) / norm(gt)
                    diff_norm = torch.linalg.norm((preds - fv_gpu).flatten(3), dim=-1)
                    rel_l2 = (diff_norm / gt_norm).mean().item()
                    
                    table_4_l2_losses[m_idx, cond_idx] += rel_l2
                    
        if chunk_idx % 100 == 99:
            print(f"  Processed {chunk_idx + 1}/{num_eval_chunks} chunks...")
            
    # Compute average
    table_4_l2_losses /= num_eval_chunks
    
    print("\n--- TABLE 4 REGENERATED (96 SENSORS - RELATIVE L2 LOSS) ---")
    print("| Model | Accuracy | Noise 5% | Noise 10% | Noise 20% | Dropout 5 | Dropout 10 | Dropout 20 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for m_idx, name in enumerate(model_names):
        r = table_4_l2_losses[m_idx]
        print("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
            name, r[0], r[1], r[2], r[3], r[4], r[5], r[6]
        ))
        
    # Save results as JSON
    results = {}
    for m_idx, name in enumerate(model_names):
        r = table_4_l2_losses[m_idx]
        results[name] = {
            "Accuracy": r[0],
            "Noise_5%": r[1],
            "Noise_10%": r[2],
            "Noise_20%": r[3],
            "Dropout_5": r[4],
            "Dropout_10": r[5],
            "Dropout_20": r[6],
        }
    with open(PROJECT_DIR_W / "table_4_robustness_96_l2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
