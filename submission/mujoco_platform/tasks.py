from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .math_utils import link_absolute_angles, forward_kinematics, wrap_to_pi


@dataclass(frozen=True)
class CircleCase:
    case_id: int
    center: tuple[float, float]
    radius: float


CIRCLE_CASES: dict[int, CircleCase] = {
    1: CircleCase(1, (0.0, 0.0), 0.5),
    2: CircleCase(2, (0.2, 0.0), 0.8),
    3: CircleCase(3, (0.0, 0.3), 0.8),
    4: CircleCase(4, (0.5, 0.5), 0.5),
}


BLUE_LENGTHS = np.array([0.4, 0.4, 0.2], dtype=float)
RED_LENGTHS = np.array([0.35, 0.45, 0.20], dtype=float)
CASE2_PERIOD = 24.0


def circle_reference(t: float, case: CircleCase, period: float = 6.0) -> dict:
    """Reference point and velocity for case 1 in the homework plane."""
    w = 2.0 * np.pi / period
    c = np.asarray(case.center, dtype=float)
    p = c + case.radius * np.array([np.cos(w * t), np.sin(w * t)])
    v = case.radius * w * np.array([-np.sin(w * t), np.cos(w * t)])
    return {
        "type": "circle",
        "case_id": case.case_id,
        "center": c,
        "radius": case.radius,
        "period": period,
        "xy": p,
        "xy_dot": v,
    }


def _case2_blue_joint_position(t: float, period: float = CASE2_PERIOD) -> np.ndarray:
    phase = 2.0 * np.pi * t / period
    radius = 0.58 + 0.28 * np.sin(3.0 * phase + 2.09) + 0.07 * np.sin(7.0 * phase + 1.1)
    radius = float(np.clip(radius, 0.25, 0.92))

    # Keep the final link radial. The first two links then solve a smooth 2-link
    # IK problem for the wrist point at radius - BLUE_LENGTHS[2].
    wrist_radius = radius - BLUE_LENGTHS[2]
    l1, l2, _ = BLUE_LENGTHS
    cos_q2 = (wrist_radius**2 - l1**2 - l2**2) / (2.0 * l1 * l2)
    cos_q2 = float(np.clip(cos_q2, -1.0, 1.0))
    q2 = float(np.arctan2(np.sqrt(max(0.0, 1.0 - cos_q2**2)), cos_q2))
    q1 = float(phase - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2)))
    q3 = float(phase - q1 - q2)
    return np.array([q1, q2, q3], dtype=float)


def blue_joint_reference(t: float) -> tuple[np.ndarray, np.ndarray]:
    """A smooth default motion for the blue robot in case 2.

    The endpoint follows a long radial Lissajous-like curve that sweeps much of
    the reachable 2D workspace while avoiding singular straight-arm motions.
    """
    q = _case2_blue_joint_position(t)
    dt = 1e-4
    qd = wrap_to_pi(_case2_blue_joint_position(t + dt) - _case2_blue_joint_position(t - dt)) / (2.0 * dt)
    return q, qd


def case2_reference(t: float) -> dict:
    q_blue, qd_blue = blue_joint_reference(t)
    blue_joints, blue_ee = forward_kinematics(BLUE_LENGTHS, q_blue)
    return {
        "type": "case2_imitation",
        "q_blue": q_blue,
        "qd_blue": qd_blue,
        "blue_joints": blue_joints,
        "blue_ee_xy": blue_ee,
        "blue_abs_angles": link_absolute_angles(q_blue),
        "red_lengths": RED_LENGTHS.copy(),
        "blue_lengths": BLUE_LENGTHS.copy(),
    }
