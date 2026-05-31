from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Sequence


@dataclass
class ArmSpec:
    name: str
    lengths: Sequence[float]
    masses: Sequence[float]
    rgba: str = "0.1 0.3 0.9 1"
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    actuated: bool = True


@dataclass
class TrailSpec:
    name: str
    count: int
    rgba: str
    size: float = 0.008


def _fmt(values) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def validate_arm_spec(spec: ArmSpec) -> None:
    if len(spec.lengths) == 0:
        raise ValueError("Each arm must contain at least one link.")
    if len(spec.lengths) != len(spec.masses):
        raise ValueError("lengths and masses must have the same length.")
    if any(float(l) <= 0 for l in spec.lengths):
        raise ValueError("All link lengths must be positive.")
    if any(float(m) <= 0 for m in spec.masses):
        raise ValueError("All endpoint masses must be positive.")


def _arm_body_xml(spec: ArmSpec) -> str:
    validate_arm_spec(spec)
    name = escape(spec.name)
    base = _fmt(spec.base_pos)
    lines: list[str] = [f'<body name="{name}_base" pos="{base}">']
    indent = "  "
    for i, (L, m) in enumerate(zip(spec.lengths, spec.masses), start=1):
        # First moving body is placed at the base; subsequent bodies are placed
        # at the previous link endpoint. The joint axis is MuJoCo y, so links
        # move in the x-z vertical plane.
        pos = "0 0 0" if i == 1 else f"{float(spec.lengths[i-2]):.8g} 0 0"
        lines.append(f'{indent}<body name="{name}_link{i}" pos="{pos}">')
        lines.append(f'{indent}  <joint name="{name}_j{i}" type="hinge" axis="0 -1 0" damping="0.04" armature="0.002" limited="false"/>')
        lines.append(f'{indent}  <geom name="{name}_rod{i}" class="rod" fromto="0 0 0 {float(L):.8g} 0 0" rgba="{spec.rgba}"/>')
        lines.append(f'{indent}  <geom name="{name}_mass{i}" type="sphere" pos="{float(L):.8g} 0 0" size="0.025" mass="{float(m):.8g}" rgba="{spec.rgba}"/>')
        if i == len(spec.lengths):
            lines.append(f'{indent}  <site name="{name}_ee" pos="{float(L):.8g} 0 0" size="0.022" rgba="1 0.2 0.1 1"/>')
        indent += "  "
    # Close link bodies and base body.
    for _ in spec.lengths:
        indent = indent[:-2]
        lines.append(f'{indent}</body>')
    lines.append('</body>')
    return "\n".join(lines)


def _actuator_xml(spec: ArmSpec, torque_limit: float) -> str:
    if not spec.actuated:
        return ""
    name = escape(spec.name)
    lines = []
    for i in range(1, len(spec.lengths) + 1):
        lines.append(f'<motor name="{name}_m{i}" joint="{name}_j{i}" gear="1" ctrllimited="true" ctrlrange="{-torque_limit:.8g} {torque_limit:.8g}"/>')
    return "\n".join(lines)


def _trail_body_xml(spec: TrailSpec) -> str:
    name = escape(spec.name)
    count = max(0, int(spec.count))
    lines = []
    for i in range(count):
        lines.append(
            f'<body name="{name}_{i}" mocap="true" pos="0 0 -10">'
            f'<geom name="{name}_{i}_geom" type="sphere" size="{float(spec.size):.8g}" '
            f'rgba="{spec.rgba}" contype="0" conaffinity="0" density="0"/>'
            '</body>'
        )
    return "\n".join(lines)


def build_model_xml(
    arms: Sequence[ArmSpec],
    timestep: float = 0.002,
    gravity: float = 9.8,
    torque_limit: float = 150.0,
    include_target_marker: bool = True,
    trail_specs: Sequence[TrailSpec] | None = None,
) -> str:
    body_xml = "\n".join(_arm_body_xml(a) for a in arms)
    actuator_xml = "\n".join(_actuator_xml(a, torque_limit) for a in arms if a.actuated)
    trail_xml = "\n".join(_trail_body_xml(spec) for spec in (trail_specs or []))
    marker_xml = """
    <body name="target_marker" mocap="true" pos="0 0 0">
      <geom name="target_marker_geom" type="sphere" size="0.028" rgba="1 0 0 0.45" contype="0" conaffinity="0" density="0"/>
    </body>
    """ if include_target_marker else ""
    return f"""
<mujoco model="planar_robot_homework">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <option timestep="{timestep:.8g}" gravity="0 0 {-float(gravity):.8g}" integrator="RK4"/>

  <default>
    <geom contype="0" conaffinity="0" friction="0 0 0"/>
    <default class="rod">
      <geom type="capsule" size="0.009" density="0" contype="0" conaffinity="0"/>
    </default>
  </default>

  <worldbody>
    <light name="top" pos="0 -3 3" dir="0 1 -1"/>
    <geom name="background" type="box" pos="0.3 0.65 0.15" size="2.8 0.01 2.1" rgba="0.92 0.92 0.92 1" contype="0" conaffinity="0" density="0"/>
    {marker_xml}
    {trail_xml}
    {body_xml}
  </worldbody>

  <actuator>
    {actuator_xml}
  </actuator>
</mujoco>
"""
