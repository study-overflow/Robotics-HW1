from __future__ import annotations

import numpy as np

from .math_utils import (
    wrap_to_pi,
    end_effector_jacobian,
    forward_kinematics,
    link_absolute_angles,
    cumulative_angle_matrix,
)


def _point_jacobian(lengths: np.ndarray, q: np.ndarray, endpoint_index: int) -> np.ndarray:
    theta = np.cumsum(q)
    n = len(lengths)
    jac = np.zeros((2, n), dtype=float)
    for col in range(endpoint_index + 1):
        link_slice = lengths[col : endpoint_index + 1]
        angle_slice = theta[col : endpoint_index + 1]
        jac[0, col] = -float(np.dot(link_slice, np.sin(angle_slice)))
        jac[1, col] = float(np.dot(link_slice, np.cos(angle_slice)))
    return jac


def _jdot_times_qd(lengths: np.ndarray, q: np.ndarray, qd: np.ndarray, endpoint_index: int | None = None) -> np.ndarray:
    endpoint_index = len(lengths) - 1 if endpoint_index is None else endpoint_index
    theta = np.cumsum(q)
    theta_dot = np.cumsum(qd)
    acc_bias = np.zeros(2, dtype=float)
    for link in range(endpoint_index + 1):
        direction = np.array([np.cos(theta[link]), np.sin(theta[link])], dtype=float)
        acc_bias -= lengths[link] * theta_dot[link] ** 2 * direction
    return acc_bias


def _mass_matrix(lengths: np.ndarray, masses: np.ndarray, q: np.ndarray, armature: float = 0.002) -> np.ndarray:
    n = len(lengths)
    M = armature * np.eye(n)
    for endpoint_index, point_mass in enumerate(masses):
        J = _point_jacobian(lengths, q, endpoint_index)
        M += point_mass * (J.T @ J)
    return 0.5 * (M + M.T)


def _coriolis_centrifugal(lengths: np.ndarray, masses: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    c = np.zeros(len(lengths), dtype=float)
    for endpoint_index, point_mass in enumerate(masses):
        J = _point_jacobian(lengths, q, endpoint_index)
        c += point_mass * (J.T @ _jdot_times_qd(lengths, q, qd, endpoint_index))
    return c


def _gravity_torque(lengths: np.ndarray, masses: np.ndarray, q: np.ndarray, gravity: float = 9.8) -> np.ndarray:
    theta = np.cumsum(q)
    g = np.zeros(len(lengths), dtype=float)
    for endpoint_index, point_mass in enumerate(masses):
        for col in range(endpoint_index + 1):
            dy_dq = float(np.dot(lengths[col : endpoint_index + 1], np.cos(theta[col : endpoint_index + 1])))
            g[col] += point_mass * gravity * dy_dq
    return g


def _damped_pinv(J: np.ndarray, damping: float) -> np.ndarray:
    task_dim = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + (damping**2) * np.eye(task_dim))


class BaseController:
    """Controller protocol used by the simulator.

    Student controllers only need to implement compute_torque(). Returning torques
    rather than desired positions keeps the platform close to the homework's
    dynamics/control objective.
    """

    def reset(self, task_info: dict) -> None:
        pass

    def compute_torque(self, t: float, q: np.ndarray, qd: np.ndarray, target: dict, info: dict) -> np.ndarray:
        raise NotImplementedError


class DemoController(BaseController):
    """Compact demo controller for both homework options.

    Case 1 is a task-space PD force controller with gravity/Coriolis
    compensation. Case 2 tracks the blue end-effector with task-space
    acceleration control and uses the Jacobian null space for posture matching.
    The implementation intentionally keeps the math close to the homework
    derivation so it is easy to read and modify.
    """

    def __init__(
        self,
        kp: float = 95.0,
        kd: float = 21.0,
        damping: float = 3.5e-2,
        force_kp: float = 900.0,
        force_kd: float = 120.0,
        null_kp: float = 8.0,
        null_kd: float = 4.0,
        max_qdd: float = 140.0,
        **_unused: object,
    ):
        self.kp = kp
        self.kd = kd
        self.damping = damping
        self.force_kp = force_kp
        self.force_kd = force_kd
        self.null_kp = null_kp
        self.null_kd = null_kd
        self.max_qdd = max_qdd

    def reset(self, task_info: dict) -> None:
        pass

    def _case1_torque(self, q: np.ndarray, qd: np.ndarray, target: dict, info: dict) -> np.ndarray:
        lengths = np.asarray(info["lengths"], dtype=float)
        masses = np.asarray(info["masses"], dtype=float)
        _, ee = forward_kinematics(lengths, q)
        x_ref = np.asarray(target["xy"], dtype=float)
        xd_ref = np.asarray(target.get("xy_dot", np.zeros(2)), dtype=float).copy()

        J = end_effector_jacobian(lengths, q)
        xdot = J @ qd
        task_force = self.force_kp * (x_ref - ee) + self.force_kd * (xd_ref - xdot)
        bias = _gravity_torque(lengths, masses, q) + _coriolis_centrifugal(lengths, masses, q, qd)
        tau = J.T @ task_force + bias
        return tau

    def _case2_torque(self, q: np.ndarray, qd: np.ndarray, target: dict, info: dict) -> np.ndarray:
        lengths = np.asarray(info["lengths"], dtype=float)
        masses = np.asarray(info["masses"], dtype=float)
        _, x = forward_kinematics(lengths, q)
        J = end_effector_jacobian(lengths, q)
        J_pinv = _damped_pinv(J, self.damping)

        x_blue = np.asarray(target["blue_ee_xy"], dtype=float)
        xd_blue = end_effector_jacobian(target["blue_lengths"], target["q_blue"]) @ target["qd_blue"]
        xdot = J @ qd
        task_acc = self.kp * (x_blue - x) + self.kd * (xd_blue - xdot)
        task_acc = np.clip(task_acc, -80.0, 80.0)
        qdd_task = J_pinv @ (task_acc - _jdot_times_qd(lengths, q, qd))

        Aang = cumulative_angle_matrix(len(q))
        pose_err = wrap_to_pi(target["blue_abs_angles"] - link_absolute_angles(q))
        qdd_pose = self.null_kp * (Aang.T @ pose_err) - self.null_kd * qd
        qdd_cmd = qdd_task + (np.eye(len(q)) - J_pinv @ J) @ qdd_pose
        qdd_cmd = np.clip(qdd_cmd, -self.max_qdd, self.max_qdd)

        M = _mass_matrix(lengths, masses, q)
        bias = _coriolis_centrifugal(lengths, masses, q, qd) + _gravity_torque(lengths, masses, q)
        tau = M @ qdd_cmd + bias
        return tau

    def compute_torque(self, t: float, q: np.ndarray, qd: np.ndarray, target: dict, info: dict) -> np.ndarray:
        if target["type"] == "circle":
            return self._case1_torque(q, qd, target, info)
        if target["type"] == "case2_imitation":
            return self._case2_torque(q, qd, target, info)
        raise ValueError(f"Unknown target type: {target['type']}")
