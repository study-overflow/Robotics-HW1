from __future__ import annotations

import numpy as np


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def cumulative_angle_matrix(n: int) -> np.ndarray:
    """Matrix A such that theta = A @ q gives absolute link angles."""
    return np.tril(np.ones((n, n), dtype=float))


def forward_kinematics(lengths: np.ndarray | list[float], q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Planar serial-chain FK in the homework plane.

    Args:
        lengths: link lengths, shape (n,).
        q: joint angles in radians, shape (n,).

    Returns:
        joints: points from base to end-effector, shape (n+1, 2). joints[0] is the base.
        ee: end-effector point, shape (2,).
    """
    lengths = np.asarray(lengths, dtype=float)
    q = np.asarray(q, dtype=float)
    theta = np.cumsum(q)
    deltas = np.column_stack([lengths * np.cos(theta), lengths * np.sin(theta)])
    joints = np.vstack([np.zeros(2), np.cumsum(deltas, axis=0)])
    return joints, joints[-1].copy()


def link_absolute_angles(q: np.ndarray) -> np.ndarray:
    return np.cumsum(np.asarray(q, dtype=float))


def end_effector_jacobian(lengths: np.ndarray | list[float], q: np.ndarray) -> np.ndarray:
    """2 x n Jacobian of the end-effector position in the homework plane."""
    lengths = np.asarray(lengths, dtype=float)
    q = np.asarray(q, dtype=float)
    theta = np.cumsum(q)
    n = len(lengths)
    j = np.zeros((2, n), dtype=float)
    for col in range(n):
        s = 0.0
        c = 0.0
        for i in range(col, n):
            s += lengths[i] * np.sin(theta[i])
            c += lengths[i] * np.cos(theta[i])
        j[0, col] = -s
        j[1, col] = c
    return j


def damped_least_squares(J: np.ndarray, error: np.ndarray, damping: float = 1e-3) -> np.ndarray:
    """Solve dq ~= argmin ||J dq - error||^2 + damping^2 ||dq||^2."""
    J = np.asarray(J, dtype=float)
    error = np.asarray(error, dtype=float)
    A = J.T @ J + (damping**2) * np.eye(J.shape[1])
    b = J.T @ error
    return np.linalg.solve(A, b)


def solve_ik_dls(
    lengths: np.ndarray | list[float],
    target_xy: np.ndarray | list[float],
    q0: np.ndarray | None = None,
    damping: float = 2e-2,
    max_iter: int = 80,
    tol: float = 1e-5,
) -> np.ndarray:
    """Small damped least-squares IK helper used by the baseline controllers."""
    lengths = np.asarray(lengths, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    q = np.zeros(len(lengths), dtype=float) if q0 is None else np.asarray(q0, dtype=float).copy()
    if np.linalg.norm(q) < 1e-10 and np.linalg.norm(target_xy) < 0.98 * float(np.sum(lengths)):
        # A fully straight arm has no first-order authority in the radial
        # direction. Start inner targets from a deterministic folded posture so
        # DLS does not falsely "converge" at the singular straight pose.
        base_seed = np.linspace(0.8, -0.4, len(lengths), dtype=float)
        if len(lengths) >= 2:
            base_seed[1:] = np.array([-1.2, 0.8, -0.4, 0.25][: len(lengths) - 1], dtype=float)
        base_seed[0] += float(np.arctan2(target_xy[1], target_xy[0]))
        q = wrap_to_pi(base_seed)
    for _ in range(max_iter):
        _, ee = forward_kinematics(lengths, q)
        err = target_xy - ee
        if np.linalg.norm(err) < tol:
            break
        J = end_effector_jacobian(lengths, q)
        dq = damped_least_squares(J, err, damping=damping)
        q = wrap_to_pi(q + np.clip(dq, -0.25, 0.25))
    return q


def homework_xy_to_mujoco_xyz(xy: np.ndarray | list[float], depth: float = 0.0) -> np.ndarray:
    """Map homework plane (x, y_vertical) to MuJoCo world (x, depth, z)."""
    xy = np.asarray(xy, dtype=float)
    return np.array([xy[0], depth, xy[1]], dtype=float)


def mujoco_xyz_to_homework_xy(xyz: np.ndarray | list[float]) -> np.ndarray:
    """Map MuJoCo world (x, depth, z) to homework plane (x, y_vertical)."""
    xyz = np.asarray(xyz, dtype=float)
    return np.array([xyz[0], xyz[2]], dtype=float)
