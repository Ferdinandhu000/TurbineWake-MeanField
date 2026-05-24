import os
from abc import ABC, abstractmethod
from typing import List, Tuple
from functools import cache

import numpy as np
import torch


class SensorGenerator(ABC):

    def __init__(
        self, 
        n_sensors: int, 
    ) -> None:
        super().__init__()
        self.n_sensors: int = n_sensors
        self.__seed: int = 0
        self.__resolution: Tuple[int, int] | None = None

    @abstractmethod
    def __call__(self) -> torch.Tensor:
        pass

    @property
    def seed(self) -> int:
        return self.__seed
    
    @seed.setter
    def seed(self, value: int) -> None:
        self.__seed = value

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.__resolution
    
    @resolution.setter
    def resolution(self, value: Tuple[int, int]) -> None:
        self.__resolution = value


class LHS(SensorGenerator):

    # implement
    @cache
    def __call__(self) -> torch.Tensor:
        assert self.resolution is not None, 'self.resolution must be set before calling a SensorGenerator'
        lhs_samples: np.ndarray = self._sampling()
        # absolute positions
        sensor_positions = torch.from_numpy(lhs_samples).cuda() * torch.tensor(data=self.resolution, device='cuda')
        # Numerical noise can occasionally push a sample to exactly the upper bound.
        # Clamp to keep indices within [0, resolution_i - 1].
        sensor_positions = sensor_positions.clamp(min=0)
        for dim, limit in enumerate(self.resolution):
            sensor_positions[:, dim].clamp_(max=limit - 1)
        return sensor_positions.int()

    def _sampling(self) -> np.ndarray:
        np.random.seed(self.seed)
        samples = np.zeros((self.n_sensors, len(self.resolution)))
        for dim in range(len(self.resolution)):
            segment_size = 1.0 / self.n_sensors
            segment_starts = np.arange(0, 1, segment_size)
            shuffled_segments: np.ndarray = np.random.permutation(segment_starts)
            for sensor in range(self.n_sensors):
                samples[sensor, dim] = np.random.uniform(
                    low=shuffled_segments[sensor], high=shuffled_segments[sensor] + segment_size
                )

        return samples
    

class AroundCylinder(SensorGenerator):

    # implement
    @cache
    def __call__(
        self, 
        hw_meters: Tuple[float, float],
        center_hw_meters: Tuple[float, float],
        radius_meters: float, 
    ) -> torch.Tensor:
        assert self.resolution is not None, 'self.resolution must be set before calling a SensorGenerator'
        assert len(self.resolution) == 2, 'AroundCylinder only works in 2D space'
        np.random.seed(self.seed)
        lhs = LHS(n_sensors=self.n_sensors); lhs.resolution = (360,)
        samples: np.ndarray = lhs._sampling().flatten()
        samples = samples * 360
        # compute meters per pixel
        h_scale: float = hw_meters[0] / self.resolution[0]
        w_scale: float = hw_meters[1] / self.resolution[1]
        # compute radius of the cylinder
        radius_w_pixels: float = radius_meters / w_scale
        radius_h_pixels: float = radius_meters / h_scale
        # compute center of the cylinder
        center_h_pixels: float = center_hw_meters[0] / h_scale
        center_w_pixels: float = center_hw_meters[1] / w_scale
        # compute sensor positions
        sensor_positions: torch.Tensor = torch.zeros((self.n_sensors, len(self.resolution)), dtype=torch.float32, device='cuda')
        sensor_positions[:, 0] = torch.from_numpy(np.cos(np.deg2rad(samples))).cuda() * radius_h_pixels + center_h_pixels
        sensor_positions[:, 1] = torch.from_numpy(np.sin(np.deg2rad(samples))).cuda() * radius_w_pixels + center_w_pixels
        sensor_positions = sensor_positions.clamp(min=0)
        sensor_positions[:, 0].clamp_(max=self.resolution[0] - 1)
        sensor_positions[:, 1].clamp_(max=self.resolution[1] - 1)
        return sensor_positions.int()



class WakeShearLHS(SensorGenerator):
    """LHS-based sensor generator that concentrates 80% of sensors in the wake
    shear-layer region (y-index in [wake_y_min, wake_y_max]) and distributes the
    remaining 20% uniformly over the full domain.

    Both sub-groups use a 1-D Latin-Hypercube scheme for their respective y-axis
    to ensure stratified, non-collapsing coverage, while x is sampled uniformly
    with LHS over the full width in both cases.
    """

    def __init__(
        self,
        n_sensors: int,
        wake_y_min: int = 25,
        wake_y_max: int = 95,
        wake_ratio: float = 0.8,
    ) -> None:
        super().__init__(n_sensors=n_sensors)
        self.wake_y_min: int = wake_y_min
        self.wake_y_max: int = wake_y_max
        self.wake_ratio: float = wake_ratio

    # implement
    @cache
    def __call__(self) -> torch.Tensor:
        assert self.resolution is not None, 'self.resolution must be set before calling a SensorGenerator'
        H, W = self.resolution

        n_wake: int  = round(self.n_sensors * self.wake_ratio)       # e.g. 102 for 128 sensors
        n_global: int = self.n_sensors - n_wake                       # e.g.  26

        rng = np.random.default_rng(self.seed)

        # ---- helper: 1-D LHS sampling in [low, high) returning n integer indices ----
        def _lhs_int(n: int, low: float, high: float, rng_: np.random.Generator) -> np.ndarray:
            """Return n integer pixel indices drawn with LHS stratification."""
            span = high - low
            seg  = span / n
            starts = low + np.arange(n) * seg
            # shuffle segment assignment
            order = rng_.permutation(n)
            positions = starts[order] + rng_.uniform(0, seg, size=n)
            return np.clip(positions, low, high - 1).astype(int)

        # ---- wake-region sensors (80%) ----
        y_wake = _lhs_int(n_wake,   float(self.wake_y_min), float(self.wake_y_max + 1), rng)
        x_wake = _lhs_int(n_wake,   0.,                     float(W),                   rng)

        # ---- global sensors (20%) ----
        y_global = _lhs_int(n_global, 0., float(H), rng)
        x_global = _lhs_int(n_global, 0., float(W), rng)

        # ---- concatenate and shuffle ----
        y_all = np.concatenate([y_wake,  y_global])
        x_all = np.concatenate([x_wake,  x_global])
        perm  = rng.permutation(self.n_sensors)
        sensor_positions = np.stack([y_all[perm], x_all[perm]], axis=1)   # shape (N, 2): (y, x)

        tensor = torch.from_numpy(sensor_positions).cuda().int()
        # clamp for safety
        tensor[:, 0].clamp_(0, H - 1)
        tensor[:, 1].clamp_(0, W - 1)
        return tensor


if __name__ == '__main__':
    resolution = (120, 256)
    lhs = LHS(n_sensors=128)
    lhs.resolution = resolution
    ac = AroundCylinder(n_sensors=128)
    ac.resolution = resolution
    ws = WakeShearLHS(n_sensors=128)
    ws.resolution = resolution
    a = lhs()
    b = ac(hw_meters=(0.14, 0.24), center_hw_meters=(0.08, 0.08), radius_meters=0.01)
    c = ws()
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    for name, sensor_positions in zip(('lhs', 'ac', 'wake_shear_lhs'), (a, b, c)):
        x_coords = sensor_positions[:, 1].detach().cpu().numpy()
        y_coords = sensor_positions[:, 0].detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlim(0, resolution[1])
        ax.set_ylim(0, resolution[0])
        if name == 'wake_shear_lhs':
            rect = patches.Rectangle(
                (0, ws.wake_y_min), resolution[1], ws.wake_y_max - ws.wake_y_min,
                linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.3,
                label=f'Wake shear layer (y={ws.wake_y_min}~{ws.wake_y_max})'
            )
            ax.add_patch(rect)
        ax.scatter(x_coords, y_coords, color='red', marker='o', s=20, label='Sensors')
        ax.legend(loc='upper right')
        ax.set_title(name)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f'{name}.png')
        plt.close(fig)
