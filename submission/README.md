# Robotics Homework 1 — Problem 1

Design a serial robot arm (≤4 DOF, total link length ≤1 m, total mass 10 kg) to optimally track four circular trajectories in the vertical plane (g=9.8 m/s²).

## Directory Structure

```
submission/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
│
├── hw_control_platform/                # MuJoCo simulation platform
│   ├── case1_controller.py             # ★ Our controller
│   ├── run_case1.py                    # Run experiments
│   ├── render_animation.py             # EGL offscreen animation renderer
│   ├── render_snapshots.py             # EGL offscreen snapshot renderer
│   └── *.py                            # Platform core (mjcf, math_utils, etc.)
│
├── reports/
│   ├── problem1_report.tex             # LaTeX source (compile with xelatex)
│   └── problem1_report.pdf             # Compiled PDF report (8 pages)
│
└── results/
    ├── animations/
    │   ├── circle1_tracking.mp4        # Circle 1 tracking animation
    │   ├── circle2_tracking.mp4        # Circle 2 tracking animation
    │   ├── circle3_tracking.mp4        # Circle 3 tracking animation
    │   └── circle4_tracking.mp4        # Circle 4 tracking animation
    └── figures/
        └── mujoco_renders/
            └── case1/
                ├── circle1_anim_grid.png
                ├── circle2_anim_grid.png
                ├── circle3_anim_grid.png
                └── circle4_anim_grid.png
```

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

Core deps: `numpy`, `scipy`, `matplotlib`, `mujoco>=3.1.0`, `pillow`.

### 2. Run Controller

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

### 3. Generate Animations

```bash
cd hw_control_platform
MUJOCO_GL=egl python render_animation.py
```

Requires `ffmpeg`. Outputs MP4 videos to `results/animations/`.

### 4. Compile Report

```bash
cd reports
xelatex problem1_report.tex
xelatex problem1_report.tex   # second pass
```

Requires a LaTeX distribution with `xelatex` and the `ctex` package.

## Controller Design

**Architecture**: Task-space PD force control with gravity and Coriolis compensation.

```
tau = J^T * [Kp*(x_ref - x) + Kd*(xdot_ref - xdot)] + G(q) + C(q,qd)
```

where Kp = 900, Kd = 60 (critically damped, ζ ≈ 1.0).

The task-space formulation directly closes the loop on end-effector error, avoiding inverse kinematics entirely. The Jacobian transpose mapping `J^T` naturally handles the 3-DOF→2D redundancy, selecting minimum-norm joint torques that satisfy the task constraint.

**Link Lengths**: [0.336, 0.338, 0.326] m — optimized via random sampling of 2000 candidate vectors, maximizing workspace reachability, manipulability, and length balance. The result is nearly equal-link, validating the classical principle that near-equal links maximize workspace dexterity.

## Results

| Circle | Center (m) | Radius (m) | RMS Error (cm) | Torque RMS (N·m) |
|--------|-----------|-----------|----------------|-------------------|
| 1      | (0.0, 0.0) | 0.5 | 0.172 | 18.9 |
| 2      | (0.2, 0.0) | 0.8 | 0.702 | 27.1 |
| 3      | (0.0, 0.3) | 0.8 | 4.376 | 26.6 |
| 4      | (0.5, 0.5) | 0.5 | 9.526 | 26.2 |
| **Avg** | | | **3.694** | **24.7** |

Circles 3 and 4 extend beyond the 1 m reachable workspace — the larger errors reflect physical limits, not controller deficiencies.
