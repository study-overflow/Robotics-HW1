"""
Problem 1 Controller: Task-Space Force Control + Model-Based Dynamics Compensation.

We use operational-space PD control (like the platform's DemoController) but with
our optimized gains and full dynamics compensation. This avoids the IK bottleneck
that joint-space computed torque suffers from, especially for large-radius circles
near workspace boundaries.

Control law:
    task_force = Kp*(x_ref - x) + Kd*(xdot_ref - xdot)
    tau = J^T * task_force + G(q) + C(q,qd)

Usage:
    python scripts/run_case1.py --controller case1_controller.py \
        --lengths 0.336 0.338 0.326 --masses 3.33 3.33 3.34 \
        --circle all --info-mode public
"""

import numpy as np
from hw_control_platform.math_utils import (
    forward_kinematics,
    end_effector_jacobian,
)
from hw_control_platform.controllers import (
    _gravity_torque,
    _coriolis_centrifugal,
)


class TaskSpaceForceController:
    """
    Task-space PD force controller with full dynamics compensation.

    Gains are tuned for the platform's large-radius circles (0.5-0.8m)
    which demand higher stiffness than our smaller circles did.
    """

    def __init__(self, task_info: dict):
        self.lengths = np.asarray(task_info.get("lengths", [0.25, 0.25, 0.25, 0.25]))
        self.masses = np.asarray(task_info.get("masses", [2.5, 2.5, 2.5, 2.5]))
        # High stiffness for large-circle tracking
        self.Kp_task = 1200.0
        self.Kd_task = 80.0

    def reset(self, task_info: dict) -> None:
        self.lengths = np.asarray(task_info.get("lengths", self.lengths))
        self.masses = np.asarray(task_info.get("masses", self.masses))

    def compute_torque(self, t: float, q: np.ndarray, qd: np.ndarray,
                       target: dict, info: dict) -> np.ndarray:
        x_ref = np.asarray(target["xy"], dtype=float)
        xd_ref = np.asarray(target.get("xy_dot", np.zeros(2)), dtype=float)

        # Forward kinematics
        _, x_current = forward_kinematics(self.lengths, q)
        J = end_effector_jacobian(self.lengths, q)
        xdot_current = J @ qd

        # Task-space PD force
        task_force = (self.Kp_task * (x_ref - x_current) +
                      self.Kd_task * (xd_ref - xdot_current))

        # Dynamics compensation
        G = _gravity_torque(self.lengths, self.masses, q)
        C = _coriolis_centrifugal(self.lengths, self.masses, q, qd)

        tau = J.T @ task_force + G + C
        return tau


def make_controller(task_info: dict):
    return TaskSpaceForceController(task_info)
