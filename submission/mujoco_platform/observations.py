from __future__ import annotations

from typing import Any

import numpy as np

from .math_utils import forward_kinematics, link_absolute_angles, wrap_to_pi


OBSERVATION_MODES = {"minimal", "state", "privileged"}
INFO_MODES = {"restricted", "public", "trusted"}


def build_public_info(
    *,
    mode: str,
    lengths: np.ndarray,
    masses: np.ndarray,
    torque_limit: float,
    timestep: float,
    qfrc_bias: np.ndarray,
    model: Any,
    data: Any,
) -> dict:
    if mode not in INFO_MODES:
        raise ValueError(f"Unknown info mode {mode!r}; expected one of {sorted(INFO_MODES)}.")
    info = {
        "lengths": np.asarray(lengths, dtype=float).copy(),
        "masses": np.asarray(masses, dtype=float).copy(),
        "torque_limit": float(torque_limit),
        "timestep": float(timestep),
        "info_mode": mode,
    }
    if mode in {"public", "trusted"}:
        info["qfrc_bias"] = np.asarray(qfrc_bias, dtype=float).copy()
    if mode == "trusted":
        info["model"] = model
        info["data"] = data
    return info


def build_observation(
    *,
    mode: str,
    task: str,
    t: float,
    q: np.ndarray,
    qd: np.ndarray,
    target: dict,
    lengths: np.ndarray,
    torque_limit: float,
    timestep: float,
    qfrc_bias: np.ndarray,
) -> dict:
    if mode not in OBSERVATION_MODES:
        raise ValueError(f"Unknown observation mode {mode!r}; expected one of {sorted(OBSERVATION_MODES)}.")
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    joints, ee = forward_kinematics(lengths, q)
    obs = {
        "mode": mode,
        "task": task,
        "t": float(t),
        "q": q.copy(),
        "qd": qd.copy(),
        "sin_q": np.sin(q),
        "cos_q": np.cos(q),
        "ee_xy": ee.copy(),
        "lengths": lengths.copy(),
        "torque_limit": float(torque_limit),
        "timestep": float(timestep),
    }
    if target["type"] == "circle":
        obs.update(
            {
                "target_xy": np.asarray(target["xy"], dtype=float).copy(),
                "target_xy_dot": np.asarray(target.get("xy_dot", np.zeros(2)), dtype=float).copy(),
                "ee_error": np.asarray(target["xy"], dtype=float) - ee,
                "circle_case_id": int(target["case_id"]),
            }
        )
    elif target["type"] == "case2_imitation":
        red_abs = link_absolute_angles(q)
        obs.update(
            {
                "target_xy": np.asarray(target["blue_ee_xy"], dtype=float).copy(),
                "target_abs_angles": np.asarray(target["blue_abs_angles"], dtype=float).copy(),
                "target_qd_hint": np.asarray(target.get("qd_blue", np.zeros_like(q)), dtype=float).copy(),
                "ee_error": np.asarray(target["blue_ee_xy"], dtype=float) - ee,
                "abs_angle_error": wrap_to_pi(np.asarray(target["blue_abs_angles"], dtype=float) - red_abs),
            }
        )
    else:
        raise ValueError(f"Unknown target type: {target['type']}")

    if mode in {"state", "privileged"}:
        obs["joint_xy"] = joints.copy()
        obs["abs_angles"] = link_absolute_angles(q)
    if mode == "privileged":
        obs["qfrc_bias"] = np.asarray(qfrc_bias, dtype=float).copy()
        if target["type"] == "case2_imitation":
            obs["q_blue"] = np.asarray(target["q_blue"], dtype=float).copy()
    return obs
