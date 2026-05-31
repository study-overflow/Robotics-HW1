#!/usr/bin/env python3
"""Render smooth MuJoCo 3D animations with glowing trails at watchable speed."""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ['MUJOCO_GL'] = 'egl'

import mujoco
import numpy as np
from scipy.interpolate import interp1d
from PIL import Image
from mjcf import ArmSpec, build_model_xml
import subprocess, shutil

N_TRAIL = 80
FPS = 25
TARGET_SECONDS = 12  # one full circle period (6s) played twice
STRIDE = 3  # render every 3rd physical step for smoothness


def build_model(lengths, masses, n_trail=N_TRAIL, w=640, h=640):
    arm = ArmSpec('robot', lengths, masses, rgba='0.9 0.18 0.12 1',
                  base_pos=(0, 0, 0), actuated=True)
    xml = build_model_xml([arm], timestep=0.002, gravity=9.8, torque_limit=150.0,
                          include_target_marker=False)

    bodies = '''
    <body name="target" mocap="true" pos="0 0 0">
      <geom name="target_geom" type="sphere" size="0.028" rgba="0 0.4 1 0.75" contype="0" conaffinity="0" density="0"/>
    </body>\n'''
    for i in range(n_trail):
        alpha = 0.30 * (1 - i / n_trail)
        bodies += f'''
    <body name="trail_t_{i}" mocap="true" pos="0 0 -10">
      <geom name="trail_t_{i}_g" type="sphere" size="0.007"
            rgba="0 0.3 1 {alpha:.3f}" contype="0" conaffinity="0" density="0"/>
    </body>'''
    for i in range(n_trail):
        alpha = 0.30 * (1 - i / n_trail)
        bodies += f'''
    <body name="trail_e_{i}" mocap="true" pos="0 0 -10">
      <geom name="trail_e_{i}_g" type="sphere" size="0.007"
            rgba="1 0.15 0.15 {alpha:.3f}" contype="0" conaffinity="0" density="0"/>
    </body>'''

    head = f'''
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
    return xml.replace('</mujoco>', head + '\n</mujoco>')


def render_animation(csv_file, output_mp4, lengths, masses):
    raw = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    n_raw = len(raw)
    t_raw = raw[:, 0]
    w, h = 640, 640

    # Interpolate data to get enough frames at our target cadence
    total_frames = TARGET_SECONDS * FPS
    t_new = np.linspace(t_raw[0], t_raw[-1], total_frames)

    interp_cols = {}
    for col, name in enumerate(['t', 'tx', 'ty', 'ex', 'ey', 'err',
                                 'q1', 'q2', 'q3', 'qd1', 'qd2', 'qd3',
                                 'tau1', 'tau2', 'tau3']):
        if col < raw.shape[1]:
            interp_cols[name] = interp1d(t_raw, raw[:, col], kind='cubic')

    xml = build_model(lengths, masses, N_TRAIL, w, h)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, w, h)

    def mocap_id(name):
        return model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]

    target_mid = mocap_id('target')
    trail_t = [mocap_id(f'trail_t_{i}') for i in range(N_TRAIL)]
    trail_e = [mocap_id(f'trail_e_{i}') for i in range(N_TRAIL)]

    def hw2mj(xy, depth=0.018):
        return np.array([xy[0], depth, xy[1]])

    trail_t_hist, trail_e_hist = [], []

    tmpdir = '/tmp/mujoco_anim_frames'
    os.makedirs(tmpdir, exist_ok=True)

    print(f'  {total_frames} frames @ {FPS}fps = {TARGET_SECONDS}s, {N_TRAIL}-pt trail...')

    for fi in range(total_frames):
        ti = t_new[fi]
        q = np.array([interp_cols[f'q{j}'](ti) for j in range(1, 1+3)])
        tx, ty = interp_cols['tx'](ti), interp_cols['ty'](ti)
        ex, ey = interp_cols['ex'](ti), interp_cols['ey'](ti)

        data.qpos[:] = q
        data.mocap_pos[target_mid] = hw2mj([tx, ty])

        trail_t_hist.append(hw2mj([tx, ty], -0.01))
        trail_e_hist.append(hw2mj([ex, ey], 0.01))
        while len(trail_t_hist) > N_TRAIL:
            trail_t_hist.pop(0); trail_e_hist.pop(0)

        for i, pos in enumerate(trail_t_hist):
            data.mocap_pos[trail_t[i]] = pos
        for i in range(len(trail_t_hist), N_TRAIL):
            data.mocap_pos[trail_t[i]] = [0, 0, -10]
        for i, pos in enumerate(trail_e_hist):
            data.mocap_pos[trail_e[i]] = pos
        for i in range(len(trail_e_hist), N_TRAIL):
            data.mocap_pos[trail_e[i]] = [0, 0, -10]

        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera='overhead')
        Image.fromarray(renderer.render()).save(
            os.path.join(tmpdir, f'frame_{fi:06d}.png'))

        if (fi + 1) % 100 == 0:
            print(f'    {fi+1}/{total_frames}')

    renderer.close()
    print(f'  Encoding...')
    subprocess.run([
        'ffmpeg', '-y', '-framerate', str(FPS),
        '-i', os.path.join(tmpdir, 'frame_%06d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23',
        output_mp4
    ], check=True, capture_output=True)
    shutil.rmtree(tmpdir)
    print(f'  -> {output_mp4}')


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(__file__))
    csv_dir = os.path.join(base, 'results', 'case1')
    out_dir = os.path.join(base, 'results', 'animations')
    os.makedirs(out_dir, exist_ok=True)

    lengths = [0.336, 0.338, 0.326]
    masses = [3.33, 3.33, 3.34]

    for cid in range(1, 5):
        f = os.path.join(csv_dir, f'case1_circle_{cid}.csv')
        if os.path.exists(f):
            print(f'\n=== Circle {cid} ===')
            render_animation(f, os.path.join(out_dir, f'circle{cid}_tracking.mp4'),
                           lengths, masses)

    print('\nDone!')
