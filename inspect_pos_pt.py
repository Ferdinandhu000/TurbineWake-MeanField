import argparse
import os
from typing import Tuple

import torch


def _as_tensor(data: object) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    raise TypeError(f"Unsupported type in pos.pt: {type(data)}")


def _coord_range(pos: torch.Tensor) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    h_min = int(pos[:, 0].min().item())
    h_max = int(pos[:, 0].max().item())
    w_min = int(pos[:, 1].min().item())
    w_max = int(pos[:, 1].max().item())
    return (h_min, h_max), (w_min, w_max)


def inspect_pos_file(path: str, head: int, save_csv: str | None) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    raw = torch.load(path, map_location="cpu", weights_only=True)
    pos = _as_tensor(raw)

    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"Expected shape (S, 2), got {tuple(pos.shape)}")

    pos = pos.int().cpu()
    num_sensors = int(pos.shape[0])
    (h_min, h_max), (w_min, w_max) = _coord_range(pos)

    unique_rows = torch.unique(pos, dim=0)
    duplicated = num_sensors - int(unique_rows.shape[0])

    print(f"Path: {path}")
    print(f"Shape: {tuple(pos.shape)}")
    print(f"Dtype: {pos.dtype}")
    print(f"Sensors: {num_sensors}")
    print(f"H range: [{h_min}, {h_max}]")
    print(f"W range: [{w_min}, {w_max}]")
    print(f"Duplicate coordinates: {duplicated}")

    n_head = max(0, min(head, num_sensors))
    if n_head > 0:
        print(f"\nFirst {n_head} coordinates as (h, w):")
        for i in range(n_head):
            h, w = pos[i].tolist()
            print(f"{i:04d}: ({h}, {w})")

    if save_csv:
        with open(save_csv, "w", encoding="utf-8") as f:
            f.write("idx,h,w\n")
            for i, (h, w) in enumerate(pos.tolist()):
                f.write(f"{i},{h},{w}\n")
        print(f"\nSaved CSV: {save_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and parse tensors sensor position file pos.pt")
    parser.add_argument(
        "--path",
        type=str,
        default="tensors/test/sensor_positions/pos.pt",
        help="Path to pos.pt file",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=20,
        help="How many coordinates to print from the top",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Optional output CSV path",
    )
    args = parser.parse_args()

    inspect_pos_file(path=args.path, head=args.head, save_csv=args.save_csv)


if __name__ == "__main__":
    main()
