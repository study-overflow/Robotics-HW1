#!/usr/bin/env python3
"""Render MuJoCo 3D animations with glowing trail behind target and end-effector."""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ['MUJOCO_GL'] = 'egl'

import mujoco
import numpy as np
from PIL import Image
from mjcf import ArmSpec, build_model_xml
import subprocess


N_TRAIL = 60  # trail length in frames


def build_model(lengths, masses, n_trail=60, w=640, h=640):
    """Build MuJoCo model with cameras, lights, target marker, and trail dots."""
    arm = ArmSpec('robot', lengths, masses, rgba='0.9 0.18 0.12 1',
                  base_pos=(0, 0, 0), actuated=True)
    xml = build_model_xml([arm], timestep=0.002, gravity=9.8, torque_limit=150.0,
                          include_target_marker=False)

    # Target marker + trail dots
    bodies = '''
    <body name="target" mocap="true" pos="0 0 0">
      <geom name="target_geom" type="sphere" size="0.025" rgba="0 0.4 1 0.7" contype="0" conaffinity="0" density="0"/>
    </body>\n'''
    for i in range(n_trail):
        bodies += f'''
    <body name="trail_t_{i}" mocap="true" pos="0 0 -10">
      <geom name="trail_t_{i}_g" type="sphere" size="0.006"
            rgba="0 0.3 1 {0.25*(1-i/n_trail):.2f}" contype="0" conaffinity="0" density="0"/>
    </body>'''
    for i in range(n_trail):
        bodies += f'''
    <body name="trail_e_{i}" mocap="true" pos="0 0 -10">
      <geom name="trail_e_{i}_g" type="sphere" size="0.006"
            rgba="1 0.2 0.2 {0.25*(1-i/n_trail):.2f}" contype="0" conaffinity="0" density="0"/>
    </body>'''

    camera_xml = f'''
  <visual>
    <global offwidth="{w}" offheight="{h}"/>
    <quality shadowsize="4096" offsamples="8"/>
  </visual>
  <statistic center="0.4 0 0.4" extent="1.2"/>
  <worldbody>
    <camera name="overhead" pos="0.5 -3 1.5" xyaxes="1 0 0 0 0.3 1" fovy="50"/>
    <light name="key"   pos="1.5 -2.5 2.5" dir="-0.3 0.8 -0.6" diffuse="0.9 0.9 0.95" specular="0.4 0.4 0.5"/>
    <light name="fill"  pos="-1 -1.5 1.5" dir="0.3 0.6 -0.5" diffuse="0.4 0.42 0.5" specular="0.1 0.1 0.15"/>
    <light name="ambient" pos="0 0 3" dir="0 0 -1" diffuse="0.25 0.27 0.3" castshadow="false"/>
    {bodies}
  </worldbody>
'''
    return xml.replace('</mujoco>', camera_xml + '\n</mujoco>')


def render_animation(csv_file, output_mp4, lengths, masses, fps=50, stride=4, w=640, h=640):
    csv_data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    n_total = len(csv_data)
    n_frames = n_total // stride
    nq = len(lengths)

    xml = build_model(lengths, masses, N_TRAIL, w, h)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, w, h)

    # Look up mocap ids
    def mocap_id(name):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return model.body_mocapid[bid]

    target_mid = mocap_id('target')
    trail_t_ids = [mocap_id(f'trail_t_{i}') for i in range(N_TRAIL)]
    trail_e_ids = [mocap_id(f'trail_e_{i}') for i in range(N_TRAIL)]

    def hw2mj(xy, depth=0.018):
        return np.array([xy[0], depth, xy[1]])

    # Pre-fill trail history
    trail_t_history = []
    trail_e_history = []

    tmpdir = '/tmp/mujoco_anim_frames'
    os.makedirs(tmpdir, exist_ok=True)

    print(f'  {n_frames} frames, {N_TRAIL}-point trail...')

    for frame_idx in range(n_total):
        t = min(frame_idx * stride, n_total - 1) if frame_idx < n_frames else n_total - 1
        q = csv_data[t, 6:6+nq]
        data.qpos[:] = q

        # Current positions
        target_xy = np.array([csv_data[t, 1], csv_data[t, 2]])
        ee_xy = np.array([csv_data[t, 3], csv_data[t, 4]])

        # Move target marker
        data.mocap_pos[target_mid] = hw2mj(target_xy)

        # Update trail histories
        trail_t_history.append(hw2mj(target_xy, -0.01))
        trail_e_history.append(hw2mj(ee_xy, 0.01))
        if len(trail_t_history) > N_TRAIL:
            trail_t_history.pop(0)
            trail_e_history.pop(0)

        # Place trail dots: oldest -> furthest back, newest -> closest to current
        for i in range(len(trail_t_history)):
            data.mocap_pos[trail_t_ids[i]] = trail_t_history[i]
        for i in range(len(trail_e_history)):
            data.mocap_pos[trail_e_ids[i]] = trail_e_history[i]
        # Hide unused trails
        for i in range(len(trail_t_history), N_TRAIL):
            data.mocap_pos[trail_t_ids[i]] = np.array([0, 0, -10])
            data.mocap_pos[trail_e_ids[i]] = np.array([0, 0, -10])

        if frame_idx % stride != 0:
            continue

        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera='overhead')
        Image.fromarray(renderer.render()).save(
            os.path.join(tmpdir, f'frame_{frame_idx//stride:06d}.png'))

        if (frame_idx // stride) % 100 == 0 and frame_idx > 0:
            print(f'    {frame_idx//stride}/{n_frames}')

    renderer.close()
    print(f'  Encoding {output_mp4}...')
    subprocess.run([
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', os.path.join(tmpdir, 'frame_%06d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23',
        output_mp4
    ], check=True, capture_output=True)

    import shutil
    shutil.rmtree(tmpdir)
    print(f'  Done: {output_mp4}')


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(__file__))
    csv_dir = os.path.join(base, 'results', 'case1')
    out_dir = os.path.join(base, 'results', 'animations')
    os.makedirs(out_dir, exist_ok=True)

    lengths = [0.336, 0.338, 0.326]
    masses = [3.33, 3.33, 3.34]

    for cid in range(1, 5):
        csv_file = os.path.join(csv_dir, f'case1_circle_{cid}.csv')
        output = os.path.join(out_dir, f'circle{cid}_tracking.mp4')
        if os.path.exists(csv_file):
            print(f'\n=== Circle {cid} ===')
            render_animation(csv_file, output, lengths, masses,
                           fps=50, stride=4, w=640, h=640)

    print('\nAll done!')
