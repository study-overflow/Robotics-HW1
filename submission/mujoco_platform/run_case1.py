#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hw_control_platform.simulator import run_case1


def parse_args():
    p = argparse.ArgumentParser(description="Run homework option 1: circle tracking.")
    p.add_argument("--controller", type=str, default=None, help="Path to a student controller .py file.")
    p.add_argument("--lengths", type=float, nargs="+", default=[0.25, 0.25, 0.25, 0.25], help="Link lengths; n<=4 and sum must be 1 m.")
    p.add_argument("--masses", type=float, nargs="+", default=[2.5, 2.5, 2.5, 2.5], help="Endpoint masses; sum must be 10 kg.")
    p.add_argument("--circle", type=str, default="all", help="Circle case id: 1, 2, 3, 4, or all.")
    p.add_argument("--duration", type=float, default=12.0)
    p.add_argument("--period", type=float, default=6.0)
    p.add_argument("--timestep", type=float, default=0.002)
    p.add_argument("--torque-limit", type=float, default=150.0)
    p.add_argument("--render", action="store_true", help="Open MuJoCo passive viewer.")
    p.add_argument("--playback-speed", type=float, default=1.0, help="Viewer playback speed multiplier; 1.0 is real time, 0.5 is half speed.")
    p.add_argument("--save-dir", type=str, default="results/case1")
    p.add_argument("--observation-mode", choices=["minimal", "state", "privileged"], default="state")
    p.add_argument("--info-mode", choices=["restricted", "public", "trusted"], default="restricted")
    return p.parse_args()


def main():
    args = parse_args()
    circle_ids = None if args.circle == "all" else [int(args.circle)]
    metrics = run_case1(
        lengths=args.lengths,
        masses=args.masses,
        controller_path=args.controller,
        circle_ids=circle_ids,
        duration=args.duration,
        period=args.period,
        timestep=args.timestep,
        torque_limit=args.torque_limit,
        render=args.render,
        playback_speed=args.playback_speed,
        save_dir=args.save_dir,
        observation_mode=args.observation_mode,
        info_mode=args.info_mode,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
