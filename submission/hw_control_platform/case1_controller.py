"""
Problem 1 Controller: Optimized Task-Space Force Control.

Key insight: the baseline uses Kd=120 with Kp=900, which gives damping ratio
ζ = Kd/(2*sqrt(Kp)) = 2.0 — heavily overdamped. This means the controller
resists velocity changes and needs larger position errors to track the circle.

Critical damping (ζ=1.0) gives the fastest convergence without overshoot.
With Kp=900, the optimal Kd = 2*sqrt(900) = 60.

Control law:
    tau = J^T * [Kp*(x_ref - x) + Kd*(xdot_ref - xdot)] + G(q) + C(q,qd)

Usage:
    python run_case1.py --controller case1_controller.py \
        --lengths 0.336 0.338 0.326 --masses 3.33 3.33 3.34 \
        --circle all --info-mode public
"""

import numpy as np
from hw_control_platform.math_utils import forward_kinematics, end_effector_jacobian
from hw_control_platform.controllers import _gravity_torque, _coriolis_centrifugal


class TaskSpaceForceController:
    """Task-space PD with critically-damped gains."""

    def __init__(self, task_info: dict):
        self.lengths = np.asarray(task_info["lengths"], dtype=float)
        self.masses  = np.asarray(task_info["masses"], dtype=float)
        # Kp=900, Kd=60 → ζ=1.0 (critical damping)
        # baseline: Kp=900, Kd=120 → ζ=2.0 (overdamped, sluggish)
        self.Kp = 900.0
        self.Kd = 60.0

    def reset(self, task_info: dict) -> None:
        self.lengths = np.asarray(task_info["lengths"], dtype=float)
        self.masses  = np.asarray(task_info["masses"], dtype=float)

    def compute_torque(self, t: float, q: np.ndarray, qd: np.ndarray,
                       target: dict, info: dict) -> np.ndarray:
        x_ref  = np.asarray(target["xy"], dtype=float)
        xd_ref = np.asarray(target.get("xy_dot", np.zeros(2)), dtype=float)

        _, x = forward_kinematics(self.lengths, q)
        J = end_effector_jacobian(self.lengths, q)
        xdot = J @ qd

        F = self.Kp * (x_ref - x) + self.Kd * (xd_ref - xdot)
        G = _gravity_torque(self.lengths, self.masses, q)
        C = _coriolis_centrifugal(self.lengths, self.masses, q, qd)
        return J.T @ F + G + C


def make_controller(task_info: dict):
    return TaskSpaceForceController(task_info)
