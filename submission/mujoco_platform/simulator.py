from __future__ import annotations

import csv
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

from controllers import DemoController
from math_utils import (
    forward_kinematics,
    link_absolute_angles,
    mujoco_xyz_to_homework_xy,
    homework_xy_to_mujoco_xyz,
    solve_ik_dls,
    wrap_to_pi,
)
from mjcf import ArmSpec, TrailSpec, build_model_xml
from observations import build_observation, build_public_info
from tasks import CIRCLE_CASES, BLUE_LENGTHS, RED_LENGTHS, circle_reference, case2_reference


def import_controller(controller_path: str | None, default_controller, task_info: dict):
    if controller_path is None:
        ctrl = default_controller()
    else:
        path = Path(controller_path)
        spec = importlib.util.spec_from_file_location("student_controller", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import controller from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "make_controller"):
            ctrl = module.make_controller(task_info)
        elif hasattr(module, "make_policy"):
            ctrl = module.make_policy(task_info)
        elif hasattr(module, "Controller"):
            ctrl = module.Controller(task_info)
        elif hasattr(module, "Policy"):
            ctrl = module.Policy(task_info)
        else:
            raise AttributeError(
                "Submission file must define make_controller(task_info), class Controller, "
                "make_policy(task_info), or class Policy."
            )
    if hasattr(ctrl, "reset"):
        ctrl.reset(task_info)
    return ctrl


def _compute_torque_from_controller(controller, t: float, q: np.ndarray, qd: np.ndarray, target: dict, info: dict, obs: dict) -> np.ndarray:
    if hasattr(controller, "compute_torque"):
        return np.asarray(controller.compute_torque(t, q, qd, target, info), dtype=float)
    if hasattr(controller, "act"):
        action = controller.act(obs)
        if isinstance(action, dict):
            action = action.get("tau", action.get("action"))
        return np.asarray(action, dtype=float)
    raise AttributeError("Submission object must implement compute_torque(...) or act(obs).")


def _require_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError(
            "MuJoCo Python package is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return mujoco


def _name_id(mujoco, model, obj_type, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise KeyError(f"MuJoCo object not found: {name}")
    return idx


def _arm_handles(mujoco, model, name: str, n: int) -> dict:
    joint_ids = [_name_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_j{i}") for i in range(1, n + 1)]
    qpos_ids = np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=int)
    qvel_ids = np.array([model.jnt_dofadr[jid] for jid in joint_ids], dtype=int)
    site_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_ee")
    act_ids = []
    for i in range(1, n + 1):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_m{i}")
        if aid >= 0:
            act_ids.append(aid)
    return {
        "joint_ids": joint_ids,
        "qpos_ids": qpos_ids,
        "qvel_ids": qvel_ids,
        "site_id": site_id,
        "act_ids": np.array(act_ids, dtype=int),
    }


def _mocap_id(mujoco, model, body_name: str) -> int:
    body_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise KeyError(f"MuJoCo body is not mocap-enabled: {body_name}")
    return mocap_id


def _trail_count(duration: float, trail_dt: float = 0.04) -> int:
    return max(2, int(np.ceil(duration / trail_dt)) + 1)


def _trail_handles(mujoco, model, name: str, count: int) -> np.ndarray:
    return np.array([_mocap_id(mujoco, model, f"{name}_{i}") for i in range(count)], dtype=int)


def _maybe_record_trail(data, trail_ids: np.ndarray | None, index: int, pos: np.ndarray) -> None:
    if trail_ids is None or index >= len(trail_ids):
        return
    data.mocap_pos[int(trail_ids[index])] = np.asarray(pos, dtype=float)


def _set_arm_state(data, handles: dict, q: np.ndarray, qd: np.ndarray | None = None):
    data.qpos[handles["qpos_ids"]] = np.asarray(q, dtype=float)
    if qd is not None:
        data.qvel[handles["qvel_ids"]] = np.asarray(qd, dtype=float)


def _get_arm_state(data, handles: dict) -> tuple[np.ndarray, np.ndarray]:
    return data.qpos[handles["qpos_ids"]].copy(), data.qvel[handles["qvel_ids"]].copy()


def _maybe_launch_viewer(render: bool, model, data):
    if not render:
        return None
    try:
        import mujoco.viewer
        return mujoco.viewer.launch_passive(model, data)
    except Exception as exc:  # pragma: no cover - viewer is platform-dependent
        print(f"[WARN] Failed to launch MuJoCo viewer: {exc}")
        return None


def _configure_planar_viewer_camera(viewer, distance: float = 3.2) -> None:
    if viewer is None:
        return
    try:
        viewer.cam.type = 0
        viewer.cam.lookat[:] = np.array([0.25, 0.0, 0.20])
        viewer.cam.distance = distance
        viewer.cam.azimuth = 90
        viewer.cam.elevation = 0
    except Exception:
        # Camera fields vary a little across MuJoCo viewer versions; rendering
        # should still work even if a version does not expose them.
        pass


def _sync_viewer(viewer, lock_planar_camera: bool = False, camera_distance: float = 3.2):
    if viewer is not None and viewer.is_running():
        if lock_planar_camera:
            _configure_planar_viewer_camera(viewer, camera_distance)
        viewer.sync()


def _save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plotting_pyplot():
    cache_dir = Path("results/.matplotlib-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir.resolve()))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _try_plot_case1(path: Path, rows: list[dict], title: str) -> None:
    try:
        plt = _plotting_pyplot()
        tx = [r["target_x"] for r in rows]
        ty = [r["target_y"] for r in rows]
        ax = [r["ee_x"] for r in rows]
        ay = [r["ee_y"] for r in rows]
        plt.figure(figsize=(5, 5))
        plt.plot(tx, ty, label="target")
        plt.plot(ax, ay, label="actual")
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Failed to save plot {path}: {exc}")


def _metrics_from_errors(errors: np.ndarray, effort: np.ndarray | None = None) -> dict:
    metrics = {
        "mean_error_m": float(np.mean(errors)),
        "rms_error_m": float(np.sqrt(np.mean(errors**2))),
        "p95_error_m": float(np.percentile(errors, 95)),
        "max_error_m": float(np.max(errors)),
    }
    if effort is not None and len(effort) > 0:
        metrics["rms_torque_Nm"] = float(np.sqrt(np.mean(effort**2)))
        metrics["max_abs_torque_Nm"] = float(np.max(np.abs(effort)))
    return metrics


def check_case1_constraints(lengths: list[float] | np.ndarray, masses: list[float] | np.ndarray, tol: float = 1e-8) -> None:
    lengths = np.asarray(lengths, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if len(lengths) > 4:
        raise ValueError("Case 1 requires no more than 4 links.")
    if abs(float(np.sum(lengths)) - 1.0) > tol:
        raise ValueError("Case 1 requires sum(lengths) == 1.0 m. Change tol if this is intentional.")
    if abs(float(np.sum(masses)) - 10.0) > tol:
        raise ValueError("Case 1 requires sum(masses) == 10.0 kg.")
    if np.any(lengths <= 0) or np.any(masses <= 0):
        raise ValueError("All lengths and endpoint masses must be positive.")


def run_case1(
    lengths: list[float] | np.ndarray,
    masses: list[float] | np.ndarray,
    controller_path: str | None = None,
    circle_ids: list[int] | None = None,
    duration: float = 12.0,
    period: float = 6.0,
    timestep: float = 0.002,
    torque_limit: float = 150.0,
    render: bool = False,
    playback_speed: float = 1.0,
    save_dir: str | Path = "results/case1",
    observation_mode: str = "state",
    info_mode: str = "restricted",
) -> dict:
    mujoco = _require_mujoco()
    lengths = np.asarray(lengths, dtype=float)
    masses = np.asarray(masses, dtype=float)
    check_case1_constraints(lengths, masses)
    if playback_speed <= 0:
        raise ValueError("playback_speed must be positive.")
    circle_ids = [1, 2, 3, 4] if circle_ids is None else circle_ids
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict] = {}
    for cid in circle_ids:
        case = CIRCLE_CASES[cid]
        arm = ArmSpec(name="robot", lengths=lengths, masses=masses, rgba="0.1 0.3 0.9 1", actuated=True)
        trail_count = _trail_count(duration)
        trail_specs = [
            TrailSpec("target_trail", trail_count, "1 0 0 0.22", size=0.006),
            TrailSpec("actual_trail", trail_count, "0.1 0.3 0.9 0.22", size=0.006),
        ] if render else None
        xml = build_model_xml(
            [arm],
            timestep=timestep,
            torque_limit=torque_limit,
            include_target_marker=True,
            trail_specs=trail_specs,
        )
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        handles = _arm_handles(mujoco, model, "robot", len(lengths))
        target_trail_ids = _trail_handles(mujoco, model, "target_trail", trail_count) if render else None
        actual_trail_ids = _trail_handles(mujoco, model, "actual_trail", trail_count) if render else None
        trail_stride = max(1, int(0.04 / timestep))
        trail_index = 0

        target0 = circle_reference(0.0, case, period=period)["xy"]
        q0 = solve_ik_dls(lengths, target0, q0=np.zeros(len(lengths)))
        _set_arm_state(data, handles, q0, np.zeros(len(lengths)))
        if model.nmocap > 0:
            data.mocap_pos[0] = homework_xy_to_mujoco_xyz(target0)
        mujoco.mj_forward(model, data)

        task_info = {
            "case": "case1_circle_tracking",
            "lengths": lengths.copy(),
            "masses": masses.copy(),
            "circle_case": case.__dict__,
            "torque_limit": torque_limit,
            "timestep": timestep,
            "observation_mode": observation_mode,
            "info_mode": info_mode,
        }
        controller = import_controller(controller_path, DemoController, task_info)
        viewer = _maybe_launch_viewer(render, model, data)
        _configure_planar_viewer_camera(viewer)
        rows: list[dict] = []
        errors = []
        efforts = []
        n_steps = int(duration / timestep)
        for step in range(n_steps):
            t = step * timestep
            target = circle_reference(t, case, period=period)
            if model.nmocap > 0:
                data.mocap_pos[0] = homework_xy_to_mujoco_xyz(target["xy"])
            mujoco.mj_forward(model, data)
            q, qd = _get_arm_state(data, handles)
            bias = data.qfrc_bias[handles["qvel_ids"]].copy()
            info = build_public_info(
                mode=info_mode,
                lengths=lengths,
                masses=masses,
                torque_limit=torque_limit,
                timestep=timestep,
                qfrc_bias=bias,
                model=model,
                data=data,
            )
            obs = build_observation(
                mode=observation_mode,
                task="case1",
                t=t,
                q=q,
                qd=qd,
                target=target,
                lengths=lengths,
                torque_limit=torque_limit,
                timestep=timestep,
                qfrc_bias=bias,
            )
            tau = _compute_torque_from_controller(controller, t, q, qd, target, info, obs)
            if tau.shape != (len(lengths),):
                raise ValueError(f"Controller returned tau shape {tau.shape}, expected {(len(lengths),)}")
            tau = np.clip(tau, -torque_limit, torque_limit)
            data.ctrl[handles["act_ids"]] = tau
            mujoco.mj_step(model, data)
            q_after, qd_after = _get_arm_state(data, handles)
            ee = mujoco_xyz_to_homework_xy(data.site_xpos[handles["site_id"]])
            err = float(np.linalg.norm(ee - target["xy"]))
            errors.append(err)
            efforts.append(tau)
            if render and step % trail_stride == 0:
                _maybe_record_trail(data, target_trail_ids, trail_index, homework_xy_to_mujoco_xyz(target["xy"], depth=-0.018))
                _maybe_record_trail(data, actual_trail_ids, trail_index, homework_xy_to_mujoco_xyz(ee, depth=0.018))
                trail_index += 1
            if step % max(1, int(0.02 / timestep)) == 0:
                rows.append({
                    "t": t,
                    "target_x": float(target["xy"][0]),
                    "target_y": float(target["xy"][1]),
                    "ee_x": float(ee[0]),
                    "ee_y": float(ee[1]),
                    "error": err,
                    **{f"q{i+1}": float(q_after[i]) for i in range(len(lengths))},
                    **{f"qd{i+1}": float(qd_after[i]) for i in range(len(lengths))},
                    **{f"tau{i+1}": float(tau[i]) for i in range(len(lengths))},
                })
            _sync_viewer(viewer, lock_planar_camera=True)
            if viewer is not None:
                time.sleep(timestep / playback_speed)
            if viewer is not None and not viewer.is_running():
                break
        if viewer is not None:
            viewer.close()
        metrics = _metrics_from_errors(np.asarray(errors), np.asarray(efforts))
        all_metrics[f"circle_{cid}"] = metrics
        _save_csv(save_dir / f"case1_circle_{cid}.csv", rows)
        _try_plot_case1(save_dir / f"case1_circle_{cid}.png", rows, f"Case 1 Circle {cid}")

    all_metrics["average"] = {
        k: float(np.mean([m[k] for key, m in all_metrics.items() if key.startswith("circle_")]))
        for k in next(iter(all_metrics.values())).keys()
    }
    with (save_dir / "metrics_case1.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    return all_metrics


def run_case2(
    controller_path: str | None = None,
    duration: float = 24.0,
    timestep: float = 0.002,
    torque_limit: float = 150.0,
    red_masses: list[float] | np.ndarray | None = None,
    render: bool = False,
    save_dir: str | Path = "results/case2",
    observation_mode: str = "state",
    info_mode: str = "restricted",
    visualization: str = "overlay",
) -> dict:
    mujoco = _require_mujoco()
    if visualization not in {"overlay", "side-by-side"}:
        raise ValueError("visualization must be 'overlay' or 'side-by-side'.")
    red_masses = np.ones(3) * (10.0 / 3.0) if red_masses is None else np.asarray(red_masses, dtype=float)
    if red_masses.shape != (3,):
        raise ValueError("red_masses must contain exactly 3 values.")
    if np.any(red_masses <= 0):
        raise ValueError("All red_masses values must be positive.")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Visualization only changes depth placement in MuJoCo. Scoring is computed
    # in each robot's local homework plane and is independent of this display mode.
    if visualization == "overlay":
        blue_base_y = -0.012
        red_base_y = 0.012
        blue_rgba = "0.05 0.25 1 0.48"
        red_rgba = "1 0.08 0.05 0.92"
    else:
        blue_base_y = -0.35
        red_base_y = 0.35
        blue_rgba = "0.05 0.25 1 1"
        red_rgba = "1 0.08 0.05 1"

    blue = ArmSpec(
        name="blue",
        lengths=BLUE_LENGTHS,
        masses=np.ones(3) * (10.0 / 3.0),
        rgba=blue_rgba,
        base_pos=(0.0, blue_base_y, 0.0),
        actuated=False,
    )
    red = ArmSpec(
        name="red",
        lengths=RED_LENGTHS,
        masses=red_masses,
        rgba=red_rgba,
        base_pos=(0.0, red_base_y, 0.0),
        actuated=True,
    )
    trail_count = _trail_count(duration)
    trail_specs = [
        TrailSpec("target_trail", trail_count, "0.05 0.25 1 0.20", size=0.006),
        TrailSpec("actual_trail", trail_count, "1 0.08 0.05 0.24", size=0.006),
    ] if render else None
    xml = build_model_xml(
        [blue, red],
        timestep=timestep,
        torque_limit=torque_limit,
        include_target_marker=True,
        trail_specs=trail_specs,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    blue_h = _arm_handles(mujoco, model, "blue", 3)
    red_h = _arm_handles(mujoco, model, "red", 3)
    target_trail_ids = _trail_handles(mujoco, model, "target_trail", trail_count) if render else None
    actual_trail_ids = _trail_handles(mujoco, model, "actual_trail", trail_count) if render else None
    trail_stride = max(1, int(0.04 / timestep))
    trail_index = 0

    target0 = case2_reference(0.0)
    _set_arm_state(data, blue_h, target0["q_blue"], target0["qd_blue"])
    q0 = solve_ik_dls(RED_LENGTHS, target0["blue_ee_xy"], q0=np.zeros(3))
    _set_arm_state(data, red_h, q0, np.zeros(3))
    if model.nmocap > 0:
        data.mocap_pos[0] = np.array([target0["blue_ee_xy"][0], red_base_y, target0["blue_ee_xy"][1]])
    mujoco.mj_forward(model, data)

    task_info = {
        "case": "case2_red_tracks_blue",
        "red_lengths": RED_LENGTHS.copy(),
        "blue_lengths": BLUE_LENGTHS.copy(),
        "red_masses": red_masses.copy(),
        "blue_masses": np.ones(3) * (10.0 / 3.0),
        "torque_limit": torque_limit,
        "timestep": timestep,
        "observation_mode": observation_mode,
        "info_mode": info_mode,
        "visualization": visualization,
        "posture_metric": "absolute_link_angle_difference_rms",
    }
    controller = import_controller(controller_path, DemoController, task_info)
    viewer = _maybe_launch_viewer(render, model, data)
    _configure_planar_viewer_camera(viewer, distance=3.0)
    rows: list[dict] = []
    ee_errors = []
    posture_errors = []
    efforts = []
    n_steps = int(duration / timestep)
    for step in range(n_steps):
        t = step * timestep
        target = case2_reference(t)
        _set_arm_state(data, blue_h, target["q_blue"], target["qd_blue"])
        if model.nmocap > 0:
            data.mocap_pos[0] = np.array([target["blue_ee_xy"][0], red_base_y, target["blue_ee_xy"][1]])
        mujoco.mj_forward(model, data)
        q, qd = _get_arm_state(data, red_h)
        bias = data.qfrc_bias[red_h["qvel_ids"]].copy()
        info = build_public_info(
            mode=info_mode,
            lengths=RED_LENGTHS,
            masses=red_masses,
            torque_limit=torque_limit,
            timestep=timestep,
            qfrc_bias=bias,
            model=model,
            data=data,
        )
        obs = build_observation(
            mode=observation_mode,
            task="case2",
            t=t,
            q=q,
            qd=qd,
            target=target,
            lengths=RED_LENGTHS,
            torque_limit=torque_limit,
            timestep=timestep,
            qfrc_bias=bias,
        )
        tau = _compute_torque_from_controller(controller, t, q, qd, target, info, obs)
        if tau.shape != (3,):
            raise ValueError(f"Controller returned tau shape {tau.shape}, expected (3,)")
        tau = np.clip(tau, -torque_limit, torque_limit)
        data.ctrl[red_h["act_ids"]] = tau
        mujoco.mj_step(model, data)

        # Restore blue exactly for logging, so the passive target does not drift.
        _set_arm_state(data, blue_h, target["q_blue"], target["qd_blue"])
        mujoco.mj_forward(model, data)
        red_q, red_qd = _get_arm_state(data, red_h)
        _, red_ee = forward_kinematics(RED_LENGTHS, red_q)
        ee_err = float(np.linalg.norm(red_ee - target["blue_ee_xy"]))
        pose_err = float(np.sqrt(np.mean(wrap_to_pi(link_absolute_angles(red_q) - target["blue_abs_angles"]) ** 2)))
        ee_errors.append(ee_err)
        posture_errors.append(pose_err)
        efforts.append(tau)
        if render and step % trail_stride == 0:
            _maybe_record_trail(
                data,
                target_trail_ids,
                trail_index,
                np.array([target["blue_ee_xy"][0], blue_base_y, target["blue_ee_xy"][1]]),
            )
            _maybe_record_trail(
                data,
                actual_trail_ids,
                trail_index,
                np.array([red_ee[0], red_base_y, red_ee[1]]),
            )
            trail_index += 1
        if step % max(1, int(0.02 / timestep)) == 0:
            rows.append({
                "t": t,
                "blue_ee_x": float(target["blue_ee_xy"][0]),
                "blue_ee_y": float(target["blue_ee_xy"][1]),
                "red_ee_x": float(red_ee[0]),
                "red_ee_y": float(red_ee[1]),
                "ee_error": ee_err,
                "posture_error_rad": pose_err,
                **{f"q_red_{i+1}": float(red_q[i]) for i in range(3)},
                **{f"q_blue_{i+1}": float(target["q_blue"][i]) for i in range(3)},
                **{f"tau_red_{i+1}": float(tau[i]) for i in range(3)},
            })
        _sync_viewer(viewer, lock_planar_camera=True, camera_distance=3.0)
        if viewer is not None and not viewer.is_running():
            break
    if viewer is not None:
        viewer.close()

    metrics = _metrics_from_errors(np.asarray(ee_errors), np.asarray(efforts))
    metrics.update({
        "mean_posture_error_rad": float(np.mean(posture_errors)),
        "rms_posture_error_rad": float(np.sqrt(np.mean(np.asarray(posture_errors) ** 2))),
        "combined_score_lower_is_better": float(
            np.sqrt(np.mean(np.asarray(ee_errors) ** 2)) + 0.15 * np.sqrt(np.mean(np.asarray(posture_errors) ** 2))
        ),
    })
    _save_csv(save_dir / "case2_tracking.csv", rows)
    try:
        plt = _plotting_pyplot()
        plt.figure(figsize=(5, 5))
        plt.plot([r["blue_ee_x"] for r in rows], [r["blue_ee_y"] for r in rows], label="blue target")
        plt.plot([r["red_ee_x"] for r in rows], [r["red_ee_y"] for r in rows], label="red actual")
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Case 2 Red tracks Blue")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / "case2_tracking.png", dpi=160)
        plt.close()
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Failed to save case2 plot: {exc}")
    with (save_dir / "metrics_case2.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics
