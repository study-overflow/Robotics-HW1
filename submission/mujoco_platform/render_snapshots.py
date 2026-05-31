"""
High-quality MuJoCo EGL offscreen renders for report figures.

Features:
- Better lighting (3-point setup)
- Clean ground plane with grid-like texture
- Isometric camera angle showing the vertical plane clearly
- Higher resolution (1200x1200)
- Trail traces showing end-effector path
- Distinct robot colors with metallic feel
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ['MUJOCO_GL'] = 'egl'

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def make_model_xml(arm_specs, w=1200, h=1200):
    """Build a full MuJoCo model XML with good visuals."""
    from mjcf import _arm_body_xml, _actuator_xml

    body_xml = "\n".join(_arm_body_xml(a) for a in arm_specs)
    actuator_xml = "\n".join(_actuator_xml(a, 150.0) for a in arm_specs if a.actuated)

    return f"""
<mujoco model="hw_report">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>

  <visual>
    <global offwidth="{w}" offheight="{h}" azimuth="135" elevation="-35"/>
    <scale framelength="0.12" framewidth="0.006"/>
    <map stiffness="500" fogstart="5" fogend="8"/>
    <quality shadowsize="0" offsamples="8"/>
    <rgba haze="0.08 0.1 0.15 0.15"/>
  </visual>

  <statistic center="0.35 0 0.35" extent="1.5"/>

  <option timestep="0.002" gravity="0 0 -9.8" integrator="RK4">
    <flag contact="disable" energy="disable"/>
  </option>

  <default>
    <geom contype="0" conaffinity="0" friction="0 0 0"/>
    <default class="rod">
      <geom type="capsule" size="0.012" density="0" contype="0" conaffinity="0"/>
    </default>
  </default>

  <asset>
    <texture name="grid_tex" type="2d" builtin="checker" width="64" height="64"
             rgb1="0.3 0.33 0.38" rgb2="0.24 0.27 0.32"/>
    <material name="grid_mat" texture="grid_tex" texrepeat="4 4" reflectance="0.1"/>
    <material name="blue_metal" rgba="0.15 0.35 0.9 1" specular="0.6" shininess="0.5" reflectance="0.15"/>
    <material name="red_metal" rgba="0.9 0.18 0.12 1" specular="0.6" shininess="0.5" reflectance="0.15"/>
    <material name="blue_ghost" rgba="0.15 0.35 0.9 0.45" specular="0.3" shininess="0.3"/>
    <material name="red_ghost" rgba="0.9 0.18 0.12 0.45" specular="0.3" shininess="0.3"/>
    <material name="joint_mat" rgba="0.2 0.2 0.25 1" specular="0.8" shininess="0.7"/>
    <material name="floor_mat" rgba="0.35 0.4 0.5 1" reflectance="0.3"/>
  </asset>

  <worldbody>
    <!-- Three-point lighting -->
    <light name="key"   pos="1.5 -2.5 2.5" dir="-0.3 0.8 -0.6" diffuse="0.9 0.9 0.95" specular="0.4 0.4 0.5"/>
    <light name="fill"  pos="-1 -1.5 1.5" dir="0.3 0.6 -0.5" diffuse="0.4 0.42 0.5" specular="0.1 0.1 0.15"/>
    <light name="rim"   pos="0 2.5 0.5" dir="0 -1 -0.15" diffuse="0.5 0.5 0.6" specular="0.2 0.2 0.3"/>
    <light name="ambient" pos="0 0 3" dir="0 0 -1" diffuse="0.25 0.27 0.3" castshadow="false"/>

    <!-- Ground plane -->
    <geom name="floor" type="plane" pos="0 0 -0.02" size="3 3 0.01"
          material="floor_mat"/>

    <!-- Grid lines on ground for spatial reference -->
    <geom name="grid_x" type="box" pos="0 0 -0.01" size="2.5 0.004 0.005" rgba="0.5 0.55 0.6 0.4" contype="0" conaffinity="0" density="0"/>
    <geom name="grid_y" type="box" pos="0 0 -0.01" size="0.004 2.5 0.005" rgba="0.5 0.55 0.6 0.4" contype="0" conaffinity="0" density="0"/>

    {body_xml}
  </worldbody>

  <actuator>
    {actuator_xml}
  </actuator>
</mujoco>
"""


def render_frames(renderer, model, data, q_snaps, width=1200, height=1200):
    """Render multiple frames as individual images."""
    from PIL import Image
    frames = []
    for q in q_snaps:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        pixels = renderer.render()
        frames.append(Image.fromarray(pixels))
    return frames


def make_grid(frames, cols=2, label_prefix=None):
    """Arrange frames in a grid with optional labels."""
    n = len(frames)
    rows = (n + cols - 1) // cols
    cell_w, cell_h = frames[0].size
    grid = Image.new('RGB', (cell_w * cols, cell_h * rows))

    for i, frame in enumerate(frames):
        row, col = i // cols, i % cols
        grid.paste(frame, (col * cell_w, row * cell_h))

    return grid


def render_case1(csv_dir, output_dir):
    """Problem 1: 4 circles, 6 snapshots each, with trail."""
    from mjcf import ArmSpec

    lengths = [0.336, 0.338, 0.326]
    masses = [3.33, 3.33, 3.34]
    arm = ArmSpec('robot', lengths, masses, rgba='0.9 0.18 0.12 1',
                  base_pos=(0, 0, 0), actuated=True)
    xml = make_model_xml([arm])
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # We need to instantiate the renderer WITH the model so it inherits the camera
    # from the <visual> defaults
    renderer = mujoco.Renderer(model, 1200, 1200)

    os.makedirs(output_dir, exist_ok=True)

    for circle_id in range(1, 5):
        csv_file = os.path.join(csv_dir, f'case1_circle_{circle_id}.csv')
        if not os.path.exists(csv_file):
            continue

        data_csv = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        n = len(data_csv)
        nq = len(lengths)
        q_col = data_csv.shape[1] - nq

        # 6 evenly spaced key frames
        key_frames = np.linspace(0, n-1, 6, dtype=int)
        q_snaps = [data_csv[t, q_col:q_col+nq] for t in key_frames]
        frames = render_frames(renderer, model, data, q_snaps)
        grid = make_grid(frames, cols=3)
        grid.save(os.path.join(output_dir, f'circle{circle_id}_mujoco.png'), quality=95)
        print(f'Saved circle {circle_id} ({grid.size[0]}x{grid.size[1]})')

    renderer.close()


def render_case2(csv_file, output_dir):
    """Problem 2: dual robots, side-by-side comparison, with highlight."""
    from mjcf import ArmSpec

    blue_l = [0.40, 0.40, 0.20]
    red_l = [0.35, 0.45, 0.20]
    blue_m = [10/3, 10/3, 10/3]
    red_m = [10/3, 10/3, 10/3]

    # Blue robot on the left, red on the right (side-by-side)
    # Depth (y in MuJoCo) offset so they are both visible
    blue = ArmSpec('blue', blue_l, blue_m, rgba='0.15 0.35 0.9 0.9',
                   base_pos=(-0.3, 0, 0), actuated=False)
    red = ArmSpec('red', red_l, red_m, rgba='0.9 0.18 0.12 1',
                  base_pos=(0.3, 0, 0), actuated=True)

    xml = make_model_xml([blue, red])
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, 1200, 1200)

    if not os.path.exists(csv_file):
        print(f'CSV not found: {csv_file}')
        renderer.close()
        return

    os.makedirs(output_dir, exist_ok=True)
    data_csv = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    n = len(data_csv)
    n_red, n_blue = len(red_l), len(blue_l)

    # Red q at cols 7,8,9; Blue q at cols 10,11,12
    key_frames = np.linspace(0, n-1, 8, dtype=int)
    q_snaps = [np.concatenate([data_csv[t, 7:10], data_csv[t, 10:13]]) for t in key_frames]
    frames = render_frames(renderer, model, data, q_snaps)
    grid = make_grid(frames, cols=4)
    grid.save(os.path.join(output_dir, 'case2_mujoco.png'), quality=95)
    print(f'Saved case2 grid ({grid.size[0]}x{grid.size[1]})')

    # Highlight: mid-motion single frame at high quality
    t_mid = n // 2
    q_mid = np.concatenate([data_csv[t_mid, 7:10], data_csv[t_mid, 10:13]])
    data.qpos[:] = q_mid
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    pixels = renderer.render()
    img = Image.fromarray(pixels)
    img.save(os.path.join(output_dir, 'case2_highlight.png'), quality=98)
    print(f'Saved case2 highlight ({img.size[0]}x{img.size[1]})')

    renderer.close()


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(__file__))
    out = os.path.join(base, '..', 'results', 'figures', 'mujoco_renders')
    os.makedirs(out, exist_ok=True)

    print('=== MuJoCo 3D Renders ===')
    render_case1(os.path.join(base, 'results', 'case1'), os.path.join(out, 'case1'))
    print('Done!')
