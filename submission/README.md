# Robotics Homework 1 — Problem 1: Single Robot Circle Tracking

## Overview

Design a serial robot arm (≤4 DOF, total link length ≤1 m, total mass 10 kg) to optimally track four circular trajectories in the vertical plane (g=9.8 m/s²).

Our solution uses:
- Optimized link lengths [0.336, 0.338, 0.326] m (3-DOF)
- Task-space force control with full dynamics compensation
- Gains: Kp=1200, Kd=80

## Directory Structure

```
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── src/                         # Core library
│   ├── robot.py                 # 3-DOF planar robot (FK, IK, dynamics)
│   ├── controller.py            # PD, CTC controllers
│   └── trajectory_generator.py  # Circle and other trajectories
├── hw_control_platform/             # MuJoCo platform integration
│   ├── case1_controller.py      # Our Problem 1 controller
│   ├── render_snapshots.py      # EGL offscreen 3D render script
│   ├── run_case1.py             # Platform runner
│   └── *.py                     # Platform core
├── configs/                     # YAML configuration
├── results/figures/             # Generated figures and renders
└── reports/                     # LaTeX source and compiled PDF
```

## Reproduction

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Our Controller

```bash
cd hw_control_platform
python run_case1.py \
    --controller case1_controller.py \
    --lengths 0.336 0.338 0.326 \
    --masses 3.33 3.33 3.34 \
    --circle all --duration 12 \
    --info-mode public \
    --save-dir ../results/case1
```

Expected average RMS error: ~3.7 cm across all four circles.

### Step 3: Generate MuJoCo 3D Renders

```bash
cd hw_control_platform
MUJOCO_GL=egl python render_snapshots.py
```

### Step 4: Compile Report

```bash
cd reports
xelatex problem1_report.tex
xelatex problem1_report.tex   # second pass for cross-references
```

### Step 5: Generate MuJoCo Animations (Optional)

```bash
cd hw_control_platform
python render_animation.py
```

Requires `ffmpeg` for video encoding. Outputs go to `results/animations/`.

## Results

| Circle | Center (m) | Radius (m) | RMS Error (cm) |
|--------|-----------|-----------|----------------|
| 1      | (0.0, 0.0) | 0.5 | 0.172 |
| 2      | (0.2, 0.0) | 0.8 | 0.702 |
| 3      | (0.0, 0.3) | 0.8 | 4.376 |
| 4      | (0.5, 0.5) | 0.5 | 9.526 |
| **Avg** | | | **3.694** |

Circles 3 and 4 extend beyond the 1m reachable workspace — the larger errors reflect physical limits, not controller limitations.

## Visual Preview

- `results/animations/` — MuJoCo 3D tracking videos (4 circles)
- `results/figures/mujoco_renders/` — MuJoCo 3D snapshot grids
- `results/figures/case1_circle_*.png` — Platform trajectory plots

## Controller Design

**Architecture**: Task-Space Force Control

```
tau = J^T * [Kp*(x_ref - x) + Kd*(xdot_ref - xdot)] + G(q) + C(q,qd)
```

The task-space formulation directly closes the loop on end-effector error, bypassing the need for inverse kinematics. This avoids the numerical instability and open-loop error accumulation that IK-based approaches suffer from near workspace boundaries.

**Link Length Optimization**: Random sampling over 2000 candidate length vectors with a composite score favoring reachability, manipulability, and length balance. The optimal [0.336, 0.338, 0.326] m is nearly equal-link, validating the classical principle that near-equal links maximize workspace dexterity.
